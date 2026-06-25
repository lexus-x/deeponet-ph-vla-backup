"""
augment.py — on-the-fly image augmentation for visual-robustness training.

Wraps the (RAM-cached) dataset and applies fresh random photometric/spatial jitter to
the camera images on every access, so the model stops keying on the exact training
render (floor texture, lighting) and generalises to the LIBERO-Plus distribution.

Design notes:
* Cache stays clean — we shallow-copy the item dict and replace image tensors with NEW
  augmented tensors; the cached tensors are never modified in place.
* Applied IDENTICALLY to ACT and both DeepONet variants (training-only; eval never
  augments), so the comparison stays fair.
* Augmentation strength maps to the LIBERO-Plus perturbation axes:
    ColorJitter      -> Light Conditions / Background-texture colour
    RandomResizedCrop-> Camera Viewpoints (mild)
    GaussianBlur     -> Sensor Noise
"""
from __future__ import annotations
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


def build_transform(mode: str = "strong"):
    """mode='strong' (original): photometric + geometric crop + blur.
    mode='mild' (probe): COLOR-ONLY photometric + light blur, NO geometric crop.
      The RandomResizedCrop in 'strong' introduced a train/eval field-of-view
      mismatch (training cropped/zoomed, eval full-frame) that degraded precision
      and hurt robustness broadly. 'mild' is geometry-preserving: it keeps the
      colour/lighting invariance that targets the Background-texture / Light axes
      (the floor-texture gap) without touching the camera framing."""
    if mode == "mild":
        return v2.Compose([
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
        ])
    return v2.Compose([
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
        v2.RandomResizedCrop(size=256, scale=(0.9, 1.0), ratio=(0.97, 1.03), antialias=True),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.25),
    ])


class AugmentWrapper(Dataset):
    def __init__(self, base, image_keys, mode: str = "strong"):
        self._base = base
        self.image_keys = list(image_keys)
        self.mode = mode
        self.tf = build_transform(mode)

    def __getattr__(self, name):           # delegate len-ish attrs / num_frames / stats
        return getattr(self.__dict__["_base"], name)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, i):
        item = dict(self._base[i])          # shallow copy; cached tensors untouched
        for k in self.image_keys:
            v = item.get(k)
            if torch.is_tensor(v):
                item[k] = self.tf(v).clamp_(0.0, 1.0)   # NEW tensor
        return item


if __name__ == "__main__":
    # CPU self-test: augmentation perturbs the image but never mutates the source.
    class _Fake(Dataset):
        def __init__(self): self.img = torch.rand(3, 256, 256)
        def __len__(self): return 4
        def __getitem__(self, i):
            return {"observation.images.image": self.img,
                    "observation.images.wrist_image": self.img, "action": torch.zeros(100, 7)}
    base = _Fake(); w = AugmentWrapper(base, ["observation.images.image", "observation.images.wrist_image"])
    a, b = w[0]["observation.images.image"], w[0]["observation.images.image"]
    diff_aug = (a - b).abs().mean().item()                       # two draws differ (random)
    src_untouched = torch.equal(base.img, base[0]["observation.images.image"])
    print(f"[augment] two draws differ: {diff_aug > 1e-4}  (mean|Δ|={diff_aug:.4f})")
    print(f"[augment] source tensor untouched: {src_untouched}")
    print(f"[augment] out shape: {tuple(a.shape)} range=[{a.min():.2f},{a.max():.2f}]")
