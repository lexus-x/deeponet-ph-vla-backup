"""
modeling_smolvla_deeponet_v2.py
===============================
SmolVLA + DeepONet head v2. Identical to v1 except the head reads the FULL prefix
token sequence via cross-attention (CrossAttnPool inside DeepONetHeadV2) instead
of a single mean-pooled context vector — fixing the spatial-localization
bottleneck. Class name SmolVLADeepONetPolicy is kept so train/evaluate import
unchanged (they point at this module in v2/).
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

from deeponet_head_v2 import DeepONetHeadV2
from ph_loss import ph_surrogate_loss
from modeling_smolvla_ph import adapt_policy_features_to_dataset  # noqa: F401

BACKBONE_PREFIX = "model.vlm_with_expert.vlm."
DEAD_PREFIXES = (
    "model.vlm_with_expert.lm_expert.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.action_time_mlp_in.",
    "model.action_time_mlp_out.",
)


class VLADeepONetV2(VLAFlowMatching):
    def __init__(self, config, p=256, d_model=512, n_queries=8, n_blocks=3, n_fourier=16,
                 head_type="deeponet"):
        super().__init__(config)
        context_dim = self.vlm_with_expert.config.text_config.hidden_size
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
        self._ph_cache = None

    def encode_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        """Run the VLM over the prefix only; return (prefix_out fp32, pad_mask)."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        (prefix_out, _), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks, position_ids=position_ids,
            past_key_values=None, inputs_embeds=[prefix_embs, None],
            use_cache=False, fill_kv_cache=True,
        )
        return prefix_out.to(torch.float32), prefix_pad_masks

    def predict_chunk(self, images, img_masks, lang_tokens, lang_masks, state) -> Tensor:
        prefix_out, pad_mask = self.encode_prefix(images, img_masks, lang_tokens, lang_masks, state)
        with torch.autocast("cuda", enabled=False):
            pred = self.deeponet(prefix_out.float(), pad_mask)
        return pred

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions,
                noise=None, time=None) -> Tensor:
        pred = self.predict_chunk(images, img_masks, lang_tokens, lang_masks, state)
        actions = actions.to(pred.dtype)
        self._ph_cache = {"pred": pred, "actions": actions}
        return F.mse_loss(pred, actions, reduction="none")

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                       noise=None, **kwargs) -> Tensor:
        return self.predict_chunk(images, img_masks, lang_tokens, lang_masks, state)


class SmolVLADeepONetPolicy(SmolVLAPolicy):
    def __init__(self, config, ph_enabled=False, lambda_ph=0.0, ph_k=8, ph_p=2.0,
                 deeponet_p=256, deeponet_blocks=3, deeponet_queries=8, deeponet_fourier=16,
                 deeponet_head="deeponet", **kwargs):
        super().__init__(config, **kwargs)
        old = self.model
        self.model = VLADeepONetV2(config, p=deeponet_p, n_blocks=deeponet_blocks,
                                   n_queries=deeponet_queries, n_fourier=deeponet_fourier,
                                   head_type=deeponet_head)
        self.model.load_state_dict(old.state_dict(), strict=False)
        del old
        self.configure_ph(ph_enabled, lambda_ph, ph_k, ph_p)
        self._freeze_dead()

    def configure_ph(self, enabled=False, lambda_ph=0.0, k=8, p=2.0):
        self.ph_enabled = bool(enabled); self.lambda_ph = float(lambda_ph)
        self.ph_k = int(k); self.ph_p = float(p); return self

    @staticmethod
    def _is_backbone(name): return name.startswith(BACKBONE_PREFIX)
    @staticmethod
    def _is_dead(name): return any(name.startswith(p) for p in DEAD_PREFIXES)

    def _freeze_dead(self):
        for n, p in self.named_parameters():
            if self._is_dead(n): p.requires_grad = False

    def forward(self, batch, noise=None, time=None, reduction="mean"):
        loss, loss_dict = super().forward(batch, noise=noise, time=time, reduction=reduction)
        cache = getattr(self.model, "_ph_cache", None)
        ph_val = loss.new_zeros(()); l1_val = loss.new_zeros(())
        if cache is not None:
            d = self.config.action_feature.shape[0]
            pred_a = cache["pred"][:, :, :d]; tgt_a = cache["actions"][:, :, :d]
            l1_val = (pred_a - tgt_a).abs().mean()
            if self.ph_enabled and self.lambda_ph > 0:
                ph_val = ph_surrogate_loss(pred_a, tgt_a, k=self.ph_k, p=self.ph_p, reduction="mean")
        total = loss + self.lambda_ph * ph_val if (self.ph_enabled and self.lambda_ph > 0) else loss
        loss_dict["flow_matching_loss"] = float(loss.detach())
        loss_dict["l1_loss"] = float(l1_val.detach())
        loss_dict["ph_loss"] = float(ph_val.detach())
        loss_dict["total_loss"] = float(total.detach())
        self.model._ph_cache = None
        return total, loss_dict

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

    def enable_gradient_checkpointing(self):
        kw = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
        try:
            self.model.vlm_with_expert.vlm.gradient_checkpointing_enable(**kw)
        except TypeError:
            self.model.vlm_with_expert.vlm.gradient_checkpointing_enable()


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = SmolVLADeepONetPolicy.from_pretrained("lerobot/smolvla_base", ph_enabled=True, lambda_ph=0.02).to(dev)
    p.freeze_backbone()
    tc = p.trainable_param_count()
    print(f"[v2] stage1 head={tc['head']/1e6:.2f}M backbone={tc['backbone']/1e6:.1f}M")
    print(f"[v2] deeponet head params = {p.model.deeponet.num_params()/1e6:.2f}M")
