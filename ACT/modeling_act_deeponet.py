"""
modeling_act_deeponet.py
========================
ACT-family policy with three heads, language conditioning, and gated PH:

  variant "act"          : stock ACT transformer decoder head  (baseline)
  variant "act_deeponet" : DeepONet operator head on the ACT encoder memory
  variant "act_deeponet_ph": same + trigger-gated persistent-homology loss

ACT has no language input, so we add a ~4M TinyLanguageEncoder whose pooled
vector is injected as ONE extra token into the ACT transformer encoder. The
CVAE structure (and its KL term) is kept for every variant; only the decoder is
swapped for the DeepONet operator in the deeponet variants.
"""
from __future__ import annotations
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from lang_encoder import TinyLanguageEncoder
from deeponet_head_v2 import DeepONetHeadV2
from ph_loss_gated import gated_ph_loss

HEADS = {"act": "transformer", "act_deeponet": "deeponet", "act_deeponet_ph": "deeponet"}


class ACTDeepONetPolicy(ACTPolicy):
    def __init__(self, config, variant: str = "act", *, lambda_ph: float = 0.02, ph_k: int = 8,
                 ph_warmup: int = 5000, ph_trigger: float = 0.15,
                 deeponet_p: int = 256, deeponet_blocks: int = 3, deeponet_queries: int = 8,
                 deeponet_fourier: int = 6, **kw):
        super().__init__(config, **kw)
        assert variant in HEADS, variant
        self.variant = variant
        self.head_kind = HEADS[variant]
        self.ph_enabled = variant.endswith("_ph")
        self.lambda_ph, self.ph_k, self.ph_warmup, self.ph_trigger = lambda_ph, ph_k, ph_warmup, ph_trigger
        d = config.dim_model
        action_dim = config.action_feature.shape[0]

        # language conditioning (one extra encoder token)
        self.lang_encoder = TinyLanguageEncoder(out_dim=d)
        self.lang_pos = nn.Parameter(torch.randn(1, d) * 0.02)

        if self.head_kind == "deeponet":
            self.deeponet_head = DeepONetHeadV2(
                context_dim=d, chunk_size=config.chunk_size, action_dim=action_dim,
                p=deeponet_p, d_model=min(d, 512), n_queries=deeponet_queries,
                n_blocks=deeponet_blocks, n_fourier=deeponet_fourier)
            # the operator head replaces ACT's decoder — drop it (and its head) so we
            # don't carry a full unused 7-layer decoder as dead weight.
            self.model.decoder = nn.Identity()
            self.model.action_head = nn.Identity()
        self._train_step = 0

    # --------------------------------------------------------------- core forward
    def _encode(self, batch, texts):
        """Replicates ACT.forward token assembly + language token; returns (memory, mu, logs)."""
        m = self.model
        cfg = self.config
        dev = batch[OBS_STATE].device
        B = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_STATE].shape[0]

        # --- CVAE latent (training only) ---
        if cfg.use_vae and ACTION in batch and self.training:
            cls = einops.repeat(m.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=B)
            vae_in = [cls]
            n_pre = 1
            if cfg.robot_state_feature:
                vae_in.append(m.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)); n_pre = 2
            vae_in.append(m.vae_encoder_action_input_proj(batch[ACTION]))
            vae_in = torch.cat(vae_in, axis=1)
            pos = m.vae_encoder_pos_enc.clone().detach()
            pad = torch.cat([torch.full((B, n_pre), False, device=dev), batch["action_is_pad"]], axis=1)
            cls_out = m.vae_encoder(vae_in.permute(1, 0, 2), pos_embed=pos.permute(1, 0, 2),
                                    key_padding_mask=pad)[0]
            pdf = m.vae_encoder_latent_output_proj(cls_out)
            mu, logs = pdf[:, : cfg.latent_dim], pdf[:, cfg.latent_dim:]
            latent = mu + logs.div(2).exp() * torch.randn_like(mu)
        else:
            mu = logs = None
            latent = torch.zeros([B, cfg.latent_dim], dtype=torch.float32, device=dev)

        # --- encoder tokens: latent, (robot_state), (env_state), images, + LANGUAGE ---
        toks = [m.encoder_latent_input_proj(latent)]
        pos = list(m.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if cfg.robot_state_feature:
            toks.append(m.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if cfg.env_state_feature:
            toks.append(m.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        # language token (token is (B,d); pos is (1,d) like ACT's other 1-D pos tokens)
        toks.append(self.lang_encoder(texts, dev))
        pos.append(self.lang_pos)
        if cfg.image_features:
            for img in batch[OBS_IMAGES]:
                cam = m.backbone(img)["feature_map"]
                cpos = m.encoder_cam_feat_pos_embed(cam).to(dtype=cam.dtype)
                cam = m.encoder_img_feat_input_proj(cam)
                toks.extend(list(einops.rearrange(cam, "b c h w -> (h w) b c")))
                pos.extend(list(einops.rearrange(cpos, "b c h w -> (h w) b c")))
        toks = torch.stack(toks, axis=0)
        pos = torch.stack(pos, axis=0)
        memory = m.encoder(toks, pos_embed=pos)               # (seq, B, d)
        return memory, mu, logs

    def _decode(self, memory, B):
        """Produce the action chunk from encoder memory via the selected head."""
        m = self.model
        if self.head_kind == "transformer":
            dec_in = torch.zeros((self.config.chunk_size, B, self.config.dim_model),
                                 dtype=memory.dtype, device=memory.device)
            dec = m.decoder(dec_in, memory, encoder_pos_embed=None,
                            decoder_pos_embed=m.decoder_pos_embed.weight.unsqueeze(1))
            dec = dec.transpose(0, 1)                          # (B, T, d)
            return m.action_head(dec)
        else:
            ctx = memory.permute(1, 0, 2)                      # (B, seq, d)
            pad_mask = torch.ones(ctx.shape[:2], dtype=torch.bool, device=ctx.device)
            return self.deeponet_head(ctx, pad_mask)           # (B, T, A)

    def _texts(self, batch):
        t = batch.get("task")
        if t is None:
            B = batch[OBS_STATE].shape[0]
            return [""] * B
        return list(t) if not isinstance(t, str) else [t]

    # --------------------------------------------------------------- train forward
    def forward(self, batch):
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        texts = self._texts(batch)
        B = batch[OBS_STATE].shape[0]
        memory, mu, logs = self._encode(batch, texts)
        actions_hat = self._decode(memory, B)

        not_pad = (~batch["action_is_pad"].unsqueeze(-1))
        l1 = (F.l1_loss(batch[ACTION], actions_hat, reduction="none") * not_pad).mean()
        info = {"l1_loss": l1.item()}
        loss = l1
        if self.config.use_vae and mu is not None:
            kld = (-0.5 * (1 + logs - mu.pow(2) - logs.exp())).sum(-1).mean()
            info["kld_loss"] = kld.item()
            loss = loss + kld * self.config.kl_weight
        if self.ph_enabled and self.training:
            ph, ph_info = gated_ph_loss(actions_hat, batch[ACTION], step=self._train_step,
                                        warmup_steps=self.ph_warmup, trigger=self.ph_trigger, k=self.ph_k)
            loss = loss + self.lambda_ph * ph
            info.update(ph_info)
            self._train_step += 1
        return loss, info

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_action_chunk(self, batch):
        self.eval()
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        texts = self._texts(batch)
        B = batch[OBS_STATE].shape[0]
        memory, _, _ = self._encode(batch, texts)
        return self._decode(memory, B)
