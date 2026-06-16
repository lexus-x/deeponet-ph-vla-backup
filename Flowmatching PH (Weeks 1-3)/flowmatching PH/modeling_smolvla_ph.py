"""
modeling_smolvla_ph.py
======================
SmolVLAPHPolicy: SmolVLA + Persistent-Homology regularization.

Subclasses the official lerobot SmolVLAPolicy (lerobot 0.5.1) and adds a
topological regularizer to the training loss:

    total_loss = flow_matching_loss + lambda_ph * PH_loss

Design notes (verified against the installed lerobot source):

* SmolVLA is a rectified-flow model. Its training forward predicts a velocity
  field v_t for x_t = t*noise + (1-t)*actions, with target u_t = noise-actions,
  and loss = MSE(u_t, v_t). There is NO explicit predicted action chunk.
  Because the flow is linear, we reconstruct the predicted clean action chunk
  in closed form from a SINGLE forward pass:

        pred_actions = x_t - t * v_t        (== actions when v_t == u_t)

  We then apply the PH surrogate between `pred_actions` and the expert
  `actions`. This is the FINAL-OUTPUT formulation (single step), NOT a
  per-denoising-step penalty -- as confirmed in the project spec.

* Backbone vs. head split (for freezing / differential LRs):
      backbone  = model.vlm_with_expert.vlm   (SmolVLM2, ~350M, pretrained)
      head      = everything else: lm_expert (action expert, ~98M) +
                  state_proj / action_in_proj / action_out_proj /
                  action_time_mlp_{in,out}  (~1.6M)

* ph_enabled=False makes this class behave EXACTLY like vanilla SmolVLA
  (no PH term computed), so the baseline and PH models share one code path.

* Inference (select_action) is inherited unchanged: PH is training-time only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    make_att_2d_masks,
)

from ph_loss import ph_surrogate_loss

# Parameter-name prefix that identifies the frozen-in-stage-1 pretrained backbone.
BACKBONE_PREFIX = "model.vlm_with_expert.vlm."


def adapt_policy_features_to_dataset(policy, ds_meta):
    """
    Re-point a pretrained SmolVLA's input/output features at a new dataset.

    smolvla_base ships with input_features = {state(6), camera1, camera2, camera3}
    and action(6). The LIBERO-10 dataset provides observation.images.image,
    observation.images.wrist_image, observation.state(8), action(7). State/action
    are internally padded to max_*_dim=32, and image keys are simply iterated, so
    we only need to overwrite the feature dicts (no weight surgery). VISUAL
    normalization is IDENTITY, so no image stats are required.

    Returns the policy (mutated in place).
    """
    from lerobot.datasets.feature_utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType

    feats = dataset_to_policy_features(ds_meta.features)
    out = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    inp = {k: v for k, v in feats.items()
           if v.type is not FeatureType.ACTION and k.startswith("observation.")}
    policy.config.input_features = inp
    policy.config.output_features = out
    return policy


class VLAFlowMatchingPH(VLAFlowMatching):
    """
    Drop-in replacement for VLAFlowMatching whose forward stashes the
    intermediate tensors (v_t, x_t, time, actions) needed to reconstruct the
    predicted action chunk for the PH loss.

    The forward body mirrors lerobot 0.5.1's VLAFlowMatching.forward exactly;
    the ONLY addition is caching `self._ph_cache`. If lerobot's internals change
    in a future version, update this method to match.
    """

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions,
                noise=None, time=None) -> Tensor:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        # --- PH addition: reconstruct predicted clean action chunk -----------
        # pred_actions = x_t - t * v_t  (closed form for rectified flow)
        # Cache in fp32 so the PH term is stable under bf16 training.
        self._ph_cache = {
            "v_t": v_t,                                  # (B, T, Dmax) fp32
            "x_t": x_t.to(torch.float32),                # (B, T, Dmax)
            "time": time.to(torch.float32),              # (B,)
            "actions": actions.to(torch.float32),        # (B, T, Dmax)
        }
        # ---------------------------------------------------------------------

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses


class SmolVLAPHPolicy(SmolVLAPolicy):
    """SmolVLA policy with optional PH regularization and training helpers."""

    # PH config (override via configure_ph or constructor kwargs)
    ph_enabled: bool = True
    lambda_ph: float = 0.1
    ph_k: int = 8
    ph_p: float = 2.0

    def __init__(self, config, ph_enabled: bool = True, lambda_ph: float = 0.1,
                 ph_k: int = 8, ph_p: float = 2.0, **kwargs):
        super().__init__(config, **kwargs)
        # Promote the flow-matching module to the PH-aware subclass in place.
        # Safe: VLAFlowMatchingPH only overrides forward + uses instance attrs.
        self.model.__class__ = VLAFlowMatchingPH
        self.model._ph_cache = None
        self.configure_ph(ph_enabled, lambda_ph, ph_k, ph_p)

    # ------------------------------------------------------------------ config
    def configure_ph(self, enabled: bool = True, lambda_ph: float = 0.1,
                     k: int = 8, p: float = 2.0) -> "SmolVLAPHPolicy":
        self.ph_enabled = bool(enabled)
        self.lambda_ph = float(lambda_ph)
        self.ph_k = int(k)
        self.ph_p = float(p)
        return self

    # ----------------------------------------------------------- training fwd
    def forward(self, batch, noise=None, time=None, reduction: str = "mean"):
        """
        Vanilla SmolVLA loss + lambda_ph * PH_loss.

        Returns (loss, loss_dict). loss_dict always includes:
          flow_matching_loss, l1_loss, ph_loss, total_loss
        so that train.py can log all four for BOTH models (PH term is 0 for the
        baseline). Behaviour for ph_enabled=False is byte-identical to vanilla
        except for the extra (zero) bookkeeping entries.
        """
        out = super().forward(batch, noise=noise, time=time, reduction=reduction)
        loss, loss_dict = out  # super returns (loss, dict) for both reductions

        cache = getattr(self.model, "_ph_cache", None)

        # Reconstruct predicted vs expert action chunks, trimmed to real dims.
        ph_val = loss.new_zeros(())
        l1_val = loss.new_zeros(())
        if cache is not None:
            v_t = cache["v_t"]
            x_t = cache["x_t"]
            t = cache["time"][:, None, None]
            tgt = cache["actions"]
            pred_actions = x_t - t * v_t  # (B, T, Dmax)

            # Trim padded action dims so topology is over real DoF only.
            d = self.config.action_feature.shape[0]
            pred_a = pred_actions[:, :, :d]
            tgt_a = tgt[:, :, :d]

            # L1 between predicted and expert chunk (logged for both models).
            l1_val = (pred_a - tgt_a).abs().mean()

            if self.ph_enabled and self.lambda_ph > 0:
                ph_val = ph_surrogate_loss(pred_a, tgt_a, k=self.ph_k, p=self.ph_p,
                                           reduction="mean")

        total = loss
        if self.ph_enabled and self.lambda_ph > 0:
            total = loss + self.lambda_ph * ph_val

        loss_dict["flow_matching_loss"] = float(loss.detach())
        loss_dict["l1_loss"] = float(l1_val.detach())
        loss_dict["ph_loss"] = float(ph_val.detach())
        loss_dict["total_loss"] = float(total.detach())

        # Free the cache so it can't leak into the next step / inference.
        self.model._ph_cache = None
        return total, loss_dict

    # ----------------------------------------------- freezing / param groups
    def _is_backbone(self, name: str) -> bool:
        return name.startswith(BACKBONE_PREFIX)

    def freeze_backbone(self) -> None:
        """Stage 1: freeze the pretrained SmolVLM2 backbone; train head only."""
        for n, p in self.named_parameters():
            p.requires_grad = not self._is_backbone(n)

    def unfreeze_all(self) -> None:
        """Stage 2: everything trainable (differential LRs applied separately)."""
        for p in self.parameters():
            p.requires_grad = True

    def param_groups(self, backbone_lr: float, head_lr: float):
        """
        Build optimizer param groups with differential LRs. Only includes
        params with requires_grad=True (so it works for Stage 1 and Stage 2).
        """
        backbone, head = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (backbone if self._is_backbone(n) else head).append(p)
        groups = []
        if head:
            groups.append({"params": head, "lr": head_lr, "name": "head"})
        if backbone:
            groups.append({"params": backbone, "lr": backbone_lr, "name": "backbone"})
        return groups

    def trainable_param_count(self):
        bb = sum(p.numel() for n, p in self.named_parameters()
                 if p.requires_grad and self._is_backbone(n))
        hd = sum(p.numel() for n, p in self.named_parameters()
                 if p.requires_grad and not self._is_backbone(n))
        return {"backbone": bb, "head": hd, "total": bb + hd}

    def enable_gradient_checkpointing(self) -> None:
        """Enable non-reentrant gradient checkpointing on backbone + expert."""
        kw = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
        for sub in (self.model.vlm_with_expert.vlm, self.model.vlm_with_expert.lm_expert):
            try:
                sub.gradient_checkpointing_enable(**kw)
            except TypeError:
                sub.gradient_checkpointing_enable()


if __name__ == "__main__":
    # Lightweight self-check: load, run one training forward, inspect loss_dict.
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[smolvla_ph] loading lerobot/smolvla_base as SmolVLAPHPolicy ...")
    policy = SmolVLAPHPolicy.from_pretrained("lerobot/smolvla_base",
                                             ph_enabled=True, lambda_ph=0.1)
    policy.to(dev)

    tc = policy.trainable_param_count()
    print(f"[smolvla_ph] (all trainable) backbone={tc['backbone']/1e6:.1f}M "
          f"head={tc['head']/1e6:.1f}M total={tc['total']/1e6:.1f}M")

    policy.freeze_backbone()
    tc = policy.trainable_param_count()
    print(f"[smolvla_ph] after freeze_backbone(): backbone={tc['backbone']/1e6:.1f}M "
          f"head={tc['head']/1e6:.1f}M  (backbone should be 0)")
    assert tc["backbone"] == 0, "freeze_backbone failed"

    pg = policy.param_groups(backbone_lr=1e-5, head_lr=1e-4)
    print("[smolvla_ph] stage-1 param groups:",
          [(g["name"], f"{sum(p.numel() for p in g['params'])/1e6:.1f}M", g["lr"]) for g in pg])

    policy.unfreeze_all()
    pg = policy.param_groups(backbone_lr=1e-5, head_lr=1e-4)
    print("[smolvla_ph] stage-2 param groups:",
          [(g["name"], f"{sum(p.numel() for p in g['params'])/1e6:.1f}M", g["lr"]) for g in pg])
    print("[smolvla_ph] OK")
