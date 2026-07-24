"""
modeling_pi05_deeponet.py
=========================
pi0.5 (lerobot `pi05`) + DeepONet operator head (+ optional PH loss).

Port of v2/modeling_smolvla_deeponet_v2.py to the pi0.5 backbone, following
DeepONet PH/PORTING_TO_PI0.md. The DeepONet head and PH loss are reused
UNCHANGED (chunk_size=50, max_action_dim=32 match between SmolVLA and pi0.5);
only the backbone names + prefix-encoding path differ.

Key structural facts (lerobot 0.5.1, policies/pi05/modeling_pi05.py):
  * Flow module  : PI05Pytorch              (analogue of VLAFlowMatching)
  * Policy        : PI05Policy(.model=PI05Pytorch)
  * Backbone      : paligemma_with_expert.paligemma (PaliGemma ~2.9B)
  * Dead expert   : paligemma_with_expert.gemma_expert + action_*_proj / time_mlp_*
  * Prefix-only   : paligemma_with_expert.forward(inputs_embeds=[prefix, None])
                    auto-routes to language_model.forward and returns
                    (prefix_out, None) — no kv-cache hack needed (cf. SmolVLA).
  * context_dim   : paligemma.config.text_config.hidden_size

NOTE/TODO(smoke-test): pi0.5's embed_prefix does NOT include proprio state in the
prefix (SmolVLA did). v1 of this head therefore conditions on image+language only.
If in-dist accuracy lags, inject a projected state token into the prefix context
(see `state_token` hook below, disabled by default) and re-time.
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    make_att_2d_masks,
)

# Reuse the EXACT head + PH loss from the SmolVLA v2 work.
_V2 = os.path.join(os.path.dirname(__file__), "..", "v2")
if _V2 not in sys.path:
    sys.path.insert(0, os.path.abspath(_V2))
from deeponet_head_v2 import DeepONetHeadV2            # noqa: E402
from ph_loss import ph_surrogate_loss                 # noqa: E402

BACKBONE_PREFIX = "model.paligemma_with_expert.paligemma."
DEAD_PREFIXES = (
    "model.paligemma_with_expert.gemma_expert.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.time_mlp_in.",
    "model.time_mlp_out.",
)


class PI05DeepONet(PI05Pytorch):
    """PI05Pytorch with the flow expert replaced by a DeepONet operator head."""

    def __init__(self, config, *, p=256, d_model=512, n_queries=8, n_blocks=3,
                 n_fourier=16, head_type="deeponet", use_state_token=False, **kw):
        super().__init__(config, **kw)
        context_dim = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        if head_type == "reg":
            from regression_head import RegressionHeadV2
            self.deeponet = RegressionHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks)
        else:
            self.deeponet = DeepONetHeadV2(
                context_dim=context_dim, chunk_size=config.chunk_size,
                action_dim=config.max_action_dim, p=p, d_model=d_model,
                n_queries=n_queries, n_blocks=n_blocks, n_fourier=n_fourier,
            )
        # Optional proprio-state token prepended to the prefix context (off by default).
        self.use_state_token = bool(use_state_token)
        if self.use_state_token:
            self.state_token = nn.Linear(config.max_state_dim, context_dim)
        self._ph_cache = None

    def encode_prefix(self, images, img_masks, tokens, masks, state=None):
        """Run PaliGemma over the prefix only; return (prefix_out fp32, pad_mask)."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        if self.use_state_token and state is not None:
            st = self.state_token(state).unsqueeze(1).to(prefix_embs.dtype)   # (B,1,D)
            prefix_embs = torch.cat([prefix_embs, st], dim=1)
            ones = torch.ones(prefix_pad_masks.shape[0], 1, dtype=prefix_pad_masks.dtype,
                              device=prefix_pad_masks.device)
            prefix_pad_masks = torch.cat([prefix_pad_masks, ones], dim=1)
            zeros = torch.zeros(prefix_att_masks.shape[0], 1, dtype=prefix_att_masks.dtype,
                                device=prefix_att_masks.device)
            prefix_att_masks = torch.cat([prefix_att_masks, zeros], dim=1)

        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d, position_ids=position_ids,
            past_key_values=None, inputs_embeds=[prefix_embs, None],
            use_cache=False, adarms_cond=[None, None],
        )
        return prefix_out.to(torch.float32), prefix_pad_masks

    def predict_chunk(self, images, img_masks, tokens, masks, state=None) -> Tensor:
        prefix_out, pad_mask = self.encode_prefix(images, img_masks, tokens, masks, state)
        with torch.autocast("cuda", enabled=False):
            pred = self.deeponet(prefix_out.float(), pad_mask)
        return pred

    def forward(self, images, img_masks, tokens, masks, actions,
                noise=None, time=None, state=None) -> Tensor:
        pred = self.predict_chunk(images, img_masks, tokens, masks, state)
        actions = actions.to(pred.dtype)
        self._ph_cache = {"pred": pred, "actions": actions}
        return F.mse_loss(pred, actions, reduction="none")

    def sample_actions(self, images, img_masks, tokens, masks, noise=None,
                       state=None, **kwargs) -> Tensor:
        return self.predict_chunk(images, img_masks, tokens, masks, state)


class PI05DeepONetPolicy(PI05Policy):
    """pi0.5 policy whose action expert is a DeepONet head; +optional PH loss.

    Mirrors SmolVLADeepONetPolicy: swap self.model, copy backbone weights with
    strict=False, freeze the dead flow-expert params, expose freeze_backbone()."""

    def __init__(self, config, *, ph_enabled=False, lambda_ph=0.0, ph_k=8, ph_p=2.0,
                 deeponet_p=256, deeponet_blocks=3, deeponet_queries=8,
                 deeponet_fourier=16, deeponet_head="deeponet",
                 use_state_token=False, **kwargs):
        super().__init__(config, **kwargs)
        old = self.model
        self.model = PI05DeepONet(
            config, p=deeponet_p, n_blocks=deeponet_blocks, n_queries=deeponet_queries,
            n_fourier=deeponet_fourier, head_type=deeponet_head,
            use_state_token=use_state_token,
            rtc_processor=getattr(old, "rtc_processor", None),
        )
        self.model.load_state_dict(old.state_dict(), strict=False)
        del old
        self.configure_ph(ph_enabled, lambda_ph, ph_k, ph_p)
        self._freeze_dead()

    def configure_ph(self, enabled=False, lambda_ph=0.0, k=8, p=2.0):
        self.ph_enabled = bool(enabled); self.lambda_ph = float(lambda_ph)
        self.ph_k = int(k); self.ph_p = float(p); return self

    def forward(self, batch, reduction="mean"):
        """Reuse PI05Policy's batch->model plumbing (sets _ph_cache), then add
        the optional PH term. Returns (total_loss, loss_dict) with the keys the
        trainer (train_pi05.py / v2 CSVLogger) expects."""
        loss, loss_dict = super().forward(batch, reduction=reduction)
        cache = getattr(self.model, "_ph_cache", None)
        ph_val = loss.new_zeros(()); l1_val = loss.new_zeros(())
        if cache is not None:
            d = self.config.output_features["action"].shape[0]
            pred_a = cache["pred"][:, :, :d]; tgt_a = cache["actions"][:, :, :d]
            l1_val = (pred_a - tgt_a).abs().mean()
            if self.ph_enabled and self.lambda_ph > 0:
                ph_val = ph_surrogate_loss(pred_a, tgt_a, k=self.ph_k, p=self.ph_p,
                                           reduction="mean")
        total = loss + self.lambda_ph * ph_val if (self.ph_enabled and self.lambda_ph > 0) else loss
        loss_dict["flow_matching_loss"] = float(loss.detach())   # = MSE here (name kept for CSV)
        loss_dict["l1_loss"] = float(l1_val.detach())
        loss_dict["ph_loss"] = float(ph_val.detach())
        loss_dict["total_loss"] = float(total.detach())
        self.model._ph_cache = None
        return total, loss_dict

    def enable_gradient_checkpointing(self):
        try:
            self.model.set_requires_grad_for_gradient_checkpointing()
        except Exception:
            pass
        self.model.gradient_checkpointing = True

    @staticmethod
    def _is_backbone(name): return name.startswith(BACKBONE_PREFIX)
    @staticmethod
    def _is_dead(name): return any(name.startswith(p) for p in DEAD_PREFIXES)

    def _freeze_dead(self):
        for n, p in self.named_parameters():
            if self._is_dead(n): p.requires_grad = False

    def freeze_backbone(self):
        for n, p in self.named_parameters():
            p.requires_grad = False if self._is_dead(n) else (not self._is_backbone(n))

    def unfreeze_all(self):
        for n, p in self.named_parameters():
            p.requires_grad = not self._is_dead(n)

    def param_groups(self, backbone_lr, head_lr):
        backbone, head = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad or self._is_dead(n): continue
            (backbone if self._is_backbone(n) else head).append(p)
        groups = []
        if head: groups.append({"params": head, "lr": head_lr, "name": "head"})
        if backbone: groups.append({"params": backbone, "lr": backbone_lr, "name": "backbone"})
        return groups

    def trainable_param_count(self):
        bb = sum(p.numel() for n, p in self.named_parameters()
                 if p.requires_grad and self._is_backbone(n) and not self._is_dead(n))
        hd = sum(p.numel() for n, p in self.named_parameters()
                 if p.requires_grad and not self._is_backbone(n) and not self._is_dead(n))
        return {"backbone": bb, "head": hd, "total": bb + hd}


class PI05FlowPolicy(PI05Policy):
    """pi0.5 FLOW baseline with the same frozen-backbone training machinery as
    PI05DeepONetPolicy, so run_stage / train_pi05.py treat all three variants
    uniformly. Backbone = PaliGemma (frozen in stage1); 'head' = the flow action
    expert (gemma_expert + action/time projections). No dead params."""

    @staticmethod
    def _is_backbone(name):
        return name.startswith(BACKBONE_PREFIX)

    @staticmethod
    def _is_dead(name):
        return False

    def freeze_backbone(self):
        for n, p in self.named_parameters():
            p.requires_grad = not self._is_backbone(n)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def param_groups(self, backbone_lr, head_lr):
        backbone, head = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (backbone if self._is_backbone(n) else head).append(p)
        groups = []
        if head: groups.append({"params": head, "lr": head_lr, "name": "head"})
        if backbone: groups.append({"params": backbone, "lr": backbone_lr, "name": "backbone"})
        return groups

    def trainable_param_count(self):
        bb = sum(p.numel() for n, p in self.named_parameters()
                 if p.requires_grad and self._is_backbone(n))
        hd = sum(p.numel() for n, p in self.named_parameters()
                 if p.requires_grad and not self._is_backbone(n))
        return {"backbone": bb, "head": hd, "total": bb + hd}

    def enable_gradient_checkpointing(self):
        try:
            self.model.set_requires_grad_for_gradient_checkpointing()
        except Exception:
            pass
        self.model.gradient_checkpointing = True

    def forward(self, batch, reduction="mean"):
        loss, ld = super().forward(batch, reduction=reduction)
        ld.setdefault("flow_matching_loss", float(loss.detach()))
        ld.setdefault("l1_loss", 0.0)
        ld.setdefault("ph_loss", 0.0)
        ld.setdefault("total_loss", float(loss.detach()))
        return loss, ld


if __name__ == "__main__":
    # Smoke test (requires pi05_base cached). Validates: weight-copy, freeze,
    # a single forward through the frozen PaliGemma + DeepONet head.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.environ.get("PI05_BASE", "lerobot/pi05_base")
    pol = PI05DeepONetPolicy.from_pretrained(ckpt, ph_enabled=True, lambda_ph=0.02).to(dev)
    pol.freeze_backbone()
    tc = pol.trainable_param_count()
    print(f"[pi05] head={tc['head']/1e6:.2f}M  backbone(trainable)={tc['backbone']/1e6:.1f}M")
    print(f"[pi05] deeponet head params = {pol.model.deeponet.num_params()/1e6:.2f}M")
