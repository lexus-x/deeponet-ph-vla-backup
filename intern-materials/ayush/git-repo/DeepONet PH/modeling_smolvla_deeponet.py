"""
modeling_smolvla_deeponet.py
============================
SmolVLA with a DeepONet action head (+ optional Persistent-Homology regularizer).

We subclass the official lerobot VLAFlowMatching / SmolVLAPolicy and swap the
iterative ~98M flow-matching action expert for the small (~2M) DeepONetHead from
deeponet_head.py.

How the backbone is reused (verified against lerobot 0.5.1 source)
-----------------------------------------------------------------
* embed_prefix(images, img_masks, lang_tokens, lang_masks, state) builds the
  multimodal prefix token sequence exactly as flow matching does (SigLIP image
  embeddings + language embeddings + projected state).
* We run the VLM text-model over the prefix ONLY by calling
      vlm_with_expert.forward(inputs_embeds=[prefix_embs, None],
                              use_cache=False, fill_kv_cache=True)
  fill_kv_cache=True forces the all-self-attention path (forward_attn_layer),
  which gracefully skips the None action stream; the lm_expert (action expert)
  is never invoked. The returned prefix hidden states are masked-mean pooled into
  a single context vector.
* DeepONetHead maps that context vector to the full action chunk in ONE pass.

Backbone / head / dead split
----------------------------
    backbone : model.vlm_with_expert.vlm.*            (SmolVLM2, ~350M, pretrained)
    head     : model.deeponet.* + model.state_proj    (trained; ~2M)
    dead     : model.vlm_with_expert.lm_expert.* and the flow-matching action
               projections (action_in_proj / action_out_proj / action_time_mlp_*)
               -- never used by the DeepONet path. Frozen and EXCLUDED from the
               optimizer and from reported head-parameter counts, so the
               comparison reflects the true active DeepONet head size. (They are
               left in the module only to avoid risky surgery on the shared
               SmolVLMWithExpertModel.)

Loss
----
    total = MSE(pred_chunk, expert_chunk) + lambda_ph * PH(pred_chunk, expert_chunk)
PH is the same differentiable top-k pairwise-distance surrogate as the
flow-matching experiments (ph_loss.py), so M3 (DeepONet) and M4 (DeepONet+PH)
differ only by the lambda_ph term -- a fair A/B.

Inference (select_action) is inherited from SmolVLAPolicy; we only override the
model's sample_actions to do a single DeepONet forward (no denoising loop).
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

from deeponet_head import DeepONetHead
from ph_loss import ph_surrogate_loss
# reuse the dataset feature adapter from the flow-matching module (identical need)
from modeling_smolvla_ph import adapt_policy_features_to_dataset  # noqa: F401

BACKBONE_PREFIX = "model.vlm_with_expert.vlm."
# Modules that exist (inherited from VLAFlowMatching) but are unused by DeepONet.
DEAD_PREFIXES = (
    "model.vlm_with_expert.lm_expert.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.action_time_mlp_in.",
    "model.action_time_mlp_out.",
)


class VLADeepONet(VLAFlowMatching):
    """VLAFlowMatching with a DeepONet head replacing the flow-matching expert."""

    def __init__(self, config, p: int = 64, branch_hidden: int = 1024,
                 trunk_hidden: int = 256, out_hidden: int = 256):
        super().__init__(config)
        context_dim = self.vlm_with_expert.config.text_config.hidden_size
        self.deeponet = DeepONetHead(
            context_dim=context_dim,
            chunk_size=config.chunk_size,
            action_dim=config.max_action_dim,
            p=p, branch_hidden=branch_hidden,
            trunk_hidden=trunk_hidden, out_hidden=out_hidden,
        )
        self._ph_cache = None  # stashes (pred, actions) for the PH term

    # --------------------------------------------------------- context encoder
    def encode_context(self, images, img_masks, lang_tokens, lang_masks, state) -> Tensor:
        """Run the VLM over the prefix only and masked-mean-pool to (B, D)."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # prefix-only pass: fill_kv_cache=True forces the self-attention path that
        # tolerates a None action stream; use_cache=False so no cache is stored.
        (prefix_out, _), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        prefix_out = prefix_out.to(torch.float32)            # (B, N, D)
        m = prefix_pad_masks.unsqueeze(-1).to(prefix_out.dtype)  # (B, N, 1)
        context = (prefix_out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # (B, D)
        return context

    def predict_chunk(self, images, img_masks, lang_tokens, lang_masks, state) -> Tensor:
        """(B, T, max_action_dim) predicted action chunk. Head runs in fp32."""
        context = self.encode_context(images, img_masks, lang_tokens, lang_masks, state)
        with torch.autocast("cuda", enabled=False):
            pred = self.deeponet(context.float())  # (B, T, A) fp32
        return pred

    # ----------------------------------------------------------- training fwd
    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions,
                noise=None, time=None) -> Tensor:
        """Return per-element MSE losses (B, T, max_action_dim), matching the
        flow-matching contract so SmolVLAPolicy.forward works unchanged."""
        pred = self.predict_chunk(images, img_masks, lang_tokens, lang_masks, state)
        actions = actions.to(pred.dtype)
        # cache (fp32) for the PH term computed in the policy wrapper
        self._ph_cache = {"pred": pred, "actions": actions}
        losses = F.mse_loss(pred, actions, reduction="none")
        return losses

    # ------------------------------------------------------------- inference
    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                       noise=None, **kwargs) -> Tensor:
        """Single-pass action chunk (no iterative denoising)."""
        return self.predict_chunk(images, img_masks, lang_tokens, lang_masks, state)


class SmolVLADeepONetPolicy(SmolVLAPolicy):
    """SmolVLA policy whose action head is a DeepONet, with optional PH loss."""

    def __init__(self, config, ph_enabled: bool = False, lambda_ph: float = 0.0,
                 ph_k: int = 8, ph_p: float = 2.0, deeponet_p: int = 64, **kwargs):
        super().__init__(config, **kwargs)
        # Replace the flow-matching model with the DeepONet model in place.
        old = self.model
        self.model = VLADeepONet(config, p=deeponet_p)
        # carry over any shared pretrained weights (state_proj + the now-dead
        # expert/projection weights) so we start from smolvla_base where possible.
        self.model.load_state_dict(old.state_dict(), strict=False)
        del old
        self.configure_ph(ph_enabled, lambda_ph, ph_k, ph_p)
        self._freeze_dead()

    # ------------------------------------------------------------------ config
    def configure_ph(self, enabled=False, lambda_ph=0.0, k=8, p=2.0):
        self.ph_enabled = bool(enabled)
        self.lambda_ph = float(lambda_ph)
        self.ph_k = int(k)
        self.ph_p = float(p)
        return self

    # --------------------------------------------------- name classification
    @staticmethod
    def _is_backbone(name: str) -> bool:
        return name.startswith(BACKBONE_PREFIX)

    @staticmethod
    def _is_dead(name: str) -> bool:
        return any(name.startswith(pfx) for pfx in DEAD_PREFIXES)

    def _freeze_dead(self):
        for n, p in self.named_parameters():
            if self._is_dead(n):
                p.requires_grad = False

    # ----------------------------------------------------------- training fwd
    def forward(self, batch, noise=None, time=None, reduction: str = "mean"):
        loss, loss_dict = super().forward(batch, noise=noise, time=time, reduction=reduction)

        cache = getattr(self.model, "_ph_cache", None)
        ph_val = loss.new_zeros(())
        l1_val = loss.new_zeros(())
        if cache is not None:
            d = self.config.action_feature.shape[0]
            pred_a = cache["pred"][:, :, :d]
            tgt_a = cache["actions"][:, :, :d]
            l1_val = (pred_a - tgt_a).abs().mean()
            if self.ph_enabled and self.lambda_ph > 0:
                ph_val = ph_surrogate_loss(pred_a, tgt_a, k=self.ph_k, p=self.ph_p,
                                           reduction="mean")

        total = loss
        if self.ph_enabled and self.lambda_ph > 0:
            total = loss + self.lambda_ph * ph_val

        # keep the same loss_dict keys as the flow-matching runs for shared logging
        loss_dict["flow_matching_loss"] = float(loss.detach())  # = MSE regression loss here
        loss_dict["l1_loss"] = float(l1_val.detach())
        loss_dict["ph_loss"] = float(ph_val.detach())
        loss_dict["total_loss"] = float(total.detach())

        self.model._ph_cache = None
        return total, loss_dict

    # ----------------------------------------------- freezing / param groups
    def freeze_backbone(self):
        """Stage 1: train DeepONet head (+state_proj) only; backbone+dead frozen."""
        for n, p in self.named_parameters():
            if self._is_dead(n):
                p.requires_grad = False
            else:
                p.requires_grad = not self._is_backbone(n)

    def unfreeze_all(self):
        """Stage 2: backbone + head trainable; dead modules stay frozen."""
        for n, p in self.named_parameters():
            p.requires_grad = not self._is_dead(n)

    def param_groups(self, backbone_lr: float, head_lr: float):
        backbone, head = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad or self._is_dead(n):
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
                 if p.requires_grad and self._is_backbone(n) and not self._is_dead(n))
        hd = sum(p.numel() for n, p in self.named_parameters()
                 if p.requires_grad and not self._is_backbone(n) and not self._is_dead(n))
        return {"backbone": bb, "head": hd, "total": bb + hd}

    def enable_gradient_checkpointing(self):
        kw = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
        try:
            self.model.vlm_with_expert.vlm.gradient_checkpointing_enable(**kw)
        except TypeError:
            self.model.vlm_with_expert.vlm.gradient_checkpointing_enable()


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[deeponet] loading lerobot/smolvla_base as SmolVLADeepONetPolicy ...")
    policy = SmolVLADeepONetPolicy.from_pretrained("lerobot/smolvla_base",
                                                   ph_enabled=True, lambda_ph=0.02)
    policy.to(dev)
    tc = policy.trainable_param_count()
    print(f"[deeponet] (all trainable) backbone={tc['backbone']/1e6:.1f}M "
          f"head={tc['head']/1e6:.2f}M")
    policy.freeze_backbone()
    tc = policy.trainable_param_count()
    print(f"[deeponet] stage1 head={tc['head']/1e6:.2f}M backbone={tc['backbone']/1e6:.1f}M "
          f"(backbone should be 0)")
    assert tc["backbone"] == 0
    deeponet_params = policy.model.deeponet.num_params()
    print(f"[deeponet] DeepONet head params = {deeponet_params/1e6:.3f}M")
    print("[deeponet] OK")
