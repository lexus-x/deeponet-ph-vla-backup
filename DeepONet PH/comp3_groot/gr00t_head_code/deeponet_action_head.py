"""
deeponet_action_head.py
=======================
DeepONet (operator-learning) action head for GR00T N1.6, a drop-in replacement
for the flow-matching diffusion `Gr00tN1d6ActionHead`. It honors the EXACT same
tensor contract the outer `Gr00tN1d6` model expects, so it can be swapped in at
`gr00t_n1d6.py` construction without touching the backbone or the training loop.

Difference from the diffusion head:
  * The diffusion head conditions a 32-layer DiT on the VLM tokens and denoises a
    noisy action trajectory over `num_inference_timesteps` steps (flow matching).
  * This head instead learns an OPERATOR: K learned queries cross-attend to the
    full VL token sequence (branch), a Fourier-featured trunk spans the action
    horizon, and their product decodes the whole action chunk in a SINGLE forward
    pass (no denoising loop). Proprio `state` is injected as one extra prefix
    token so the operator is conditioned on it.

Loss: masked MSE regression against the expert action chunk (same masking as the
diffusion head via `action_input.action_mask`). With `GR00T_PH=1` a differentiable
persistent-homology surrogate is added over the VALID action dims only.

Swap is env-driven (see gr00t_n1d6.py):
  GR00T_ACTION_HEAD = diffusion | deeponet | deeponet_ph   (default: diffusion)
  GR00T_PH          = 0 | 1        (set to 1 for the +PH variant)
  GR00T_LAMBDA_PH   = float        (PH loss weight, default 0.02)

Backbone freezing is handled upstream (tune_llm/tune_visual=False on the backbone);
this head is fully trainable — it IS the module we are comparing.
"""

from __future__ import annotations

import os

import torch
from torch import nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

from .deeponet_head_v2 import DeepONetHeadV2
from .ph_loss import ph_surrogate_loss


class Gr00tN1d6DeepONetActionHead(nn.Module):
    """Operator action head with the Gr00tN1d6ActionHead interface."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.action_dim = config.max_action_dim          # 29 (padded); LIBERO uses 7
        self.action_horizon = config.action_horizon       # 16
        self.state_dim = config.max_state_dim              # 29 (padded)
        bb = config.backbone_embedding_dim                 # 2048

        # LayerNorm on VL tokens (name matches the diffusion head so the pretrained
        # vlln weights load cleanly; harmless if they don't).
        self.vlln = nn.LayerNorm(bb) if config.use_vlln else nn.Identity()
        # Proprioceptive state -> one extra VL-space token conditioning the operator.
        self.state_proj = nn.Linear(self.state_dim, bb)

        # The operator: branch cross-attends VL+state tokens, trunk spans the horizon.
        self.deeponet = DeepONetHeadV2(
            context_dim=bb,
            chunk_size=self.action_horizon,
            action_dim=self.action_dim,
        )

        # PH regularizer config (env-driven so the runner can toggle per variant).
        self.ph_enabled = os.environ.get("GR00T_PH", "0") == "1"
        self.lambda_ph = float(os.environ.get("GR00T_LAMBDA_PH", "0.02"))

        # Interface parity flags (this head is fully trainable regardless).
        self.tune_projector = True
        self.tune_diffusion_model = True
        self.tune_vlln = bool(getattr(config, "tune_vlln", True))

    # ---- interface parity methods (called by the outer model / trainer) --------
    def set_trainable_parameters(self, tune_projector=True, tune_diffusion_model=True,
                                 tune_vlln=True):
        """The operator head is the module under test -> keep it trainable. Only
        honor tune_vlln so behavior matches the diffusion head's vlln handling."""
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_vlln and isinstance(self.vlln, nn.LayerNorm):
            self.vlln.requires_grad_(False)
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[GR00T-DeepONet] trainable head params: {n/1e6:.2f}M "
              f"(PH={self.ph_enabled}, lambda_ph={self.lambda_ph})")

    def set_frozen_modules_to_eval_mode(self):
        if self.training and not self.tune_vlln and isinstance(self.vlln, nn.LayerNorm):
            self.vlln.eval()

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_output["backbone_features"] = self.vlln(backbone_output["backbone_features"])
        return backbone_output

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    # ---- core ------------------------------------------------------------------
    def _prefix(self, backbone_output: BatchFeature, action_input: BatchFeature):
        """Build the operator input function: VL tokens (+ one state token) and a
        validity mask (True = keep). Applies vlln here (mirrors the diffusion head
        applying it before the DiT)."""
        vl = self.vlln(backbone_output.backbone_features)            # (B,S,bb)
        mask = backbone_output.backbone_attention_mask              # (B,S) 1=valid
        # GR00T state is a (B, T_state, state_dim) sequence (T_state may be 1), NOT a
        # single vector -> Linear keeps the time dim, giving (B, T_state, bb). Append
        # those state tokens to the VL sequence (mirrors the diffusion head, which also
        # treats state_features as a (B,T,D) token sequence).
        st = self.state_proj(action_input.state.to(vl.dtype))       # (B,T_state,bb)
        if st.dim() == 2:                                           # (B,bb) -> (B,1,bb) defensive
            st = st.unsqueeze(1)
        n_st = st.shape[1]
        prefix = torch.cat([vl, st], dim=1)                        # (B,S+T_state,bb)
        ones = torch.ones(mask.shape[0], n_st, dtype=mask.dtype, device=mask.device)
        pad = torch.cat([mask, ones], dim=1).bool()                # (B,S+T_state) True=valid
        return prefix, pad

    def set_frozen_modules_to_eval_mode_safe(self):  # defensive alias
        self.set_frozen_modules_to_eval_mode()

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        prefix, pad = self._prefix(backbone_output, action_input)
        pred = self.deeponet(prefix, pad)                           # (B,T,A)

        target = action_input.action                               # (B,T,A)
        amask = action_input.action_mask                           # (B,T,A) float
        mse = F.mse_loss(pred, target, reduction="none") * amask
        loss = mse.sum() / (amask.sum() + 1e-6)


        if self.ph_enabled:
            # Topology only over the physically-valid action dims (padded dims are
            # constant zeros and would distort the point-cloud spectrum).
            vd = amask[0, 0].bool()                                # (A,)
            if vd.any():
                ph = ph_surrogate_loss(pred[..., vd], target[..., vd])
                loss = loss + self.lambda_ph * ph

        return {
            "loss": loss,
            "action_loss": mse,
            "action_mask": amask,
            "backbone_features": backbone_output.backbone_features,
            "state_features": None,
        }

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        prefix, pad = self._prefix(backbone_output, action_input)
        pred = self.deeponet(prefix, pad)                           # (B,T,A)
        return BatchFeature(data={"action_pred": pred})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
