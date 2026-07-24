#!/usr/bin/env python
"""Build PPT support assets: architecture diagram, a side-by-side PH-vs-flow video
(where PH succeeds and flow-matching fails), and representative frames."""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import imageio.v2 as imageio
import cv2

OUT = Path("hugging face ckp/ppt_assets"); OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- architecture
def architecture():
    fig, ax = plt.subplots(figsize=(13, 7)); ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 7)

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc=color, ec="black", lw=1.5))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10, weight="bold")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=18, lw=1.8, color="#333"))

    # inputs
    box(0.3, 5.3, 2.4, 0.9, "Camera 1\n(agentview)", "#cfe8ff")
    box(0.3, 4.1, 2.4, 0.9, "Camera 2\n(wrist)", "#cfe8ff")
    box(0.3, 2.9, 2.4, 0.9, "Robot state\n(8-dim)", "#d7f0d7")
    box(0.3, 1.7, 2.4, 0.9, "Language\ninstruction", "#ffe6b3")
    # backbone
    box(3.5, 3.3, 3.0, 2.4, "SmolVLM-2\nVision-Language\nBackbone (~350M)", "#e8d5ff")
    # expert
    box(7.2, 3.6, 2.6, 1.8, "Action Expert\n(flow matching,\n~100M)", "#ffd5d5")
    # output
    box(10.3, 3.8, 2.4, 1.4, "Predicted\naction chunk\n(50 x 7)", "#fff0a8")
    # losses
    box(7.2, 1.3, 2.6, 1.2, "Flow-matching\nloss", "#d5d5d5")
    box(10.3, 1.3, 2.4, 1.2, "PH loss\n(topology of\naction chunk)", "#ffb3b3")

    for y in (5.75, 4.55, 3.35, 2.15):
        arrow(2.7, y, 3.5, 4.5)
    arrow(6.5, 4.5, 7.2, 4.5)
    arrow(9.8, 4.5, 10.3, 4.5)
    arrow(8.5, 3.6, 8.5, 2.5)            # expert -> flow loss
    arrow(11.5, 3.8, 11.5, 2.5)          # action chunk -> PH loss
    arrow(10.3, 1.9, 9.8, 1.9)           # PH compares against expert chunk
    ax.text(6.5, 0.7, r"$L_{total} = L_{flow\ matching} + \lambda \cdot L_{PH}$    (λ = 0.1)",
            ha="center", fontsize=14, weight="bold",
            bbox=dict(boxstyle="round", fc="#fffbe6", ec="black"))
    ax.text(6.5, 6.6, "SmolVLA + Persistent-Homology regularization (training-time only)",
            ha="center", fontsize=13, weight="bold")
    fig.savefig(OUT / "architecture.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("architecture.png")


# ---------------------------------------------------------------- video + frames
def read_frames(path, maxf=200):
    r = imageio.get_reader(path)
    fr = []
    for i, f in enumerate(r):
        if i >= maxf: break
        fr.append(np.asarray(f)[:, :, :3])
    r.close()
    return fr


def side_by_side_video(base_mp4, ph_mp4, out_mp4, frames_png):
    fb = read_frames(base_mp4); fp = read_frames(ph_mp4)
    n = max(len(fb), len(fp))
    H = 256
    def prep(f, label, color):
        f = cv2.resize(f, (H, H))
        f = cv2.copyMakeBorder(f, 28, 6, 6, 6, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(f, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return f
    out = []
    for i in range(n):
        a = prep(fb[min(i, len(fb)-1)], "Flow-matching: FAIL", (80, 80, 255))
        b = prep(fp[min(i, len(fp)-1)], "Flow + PH: SUCCESS", (80, 255, 80))
        sep = np.full((a.shape[0], 6, 3), 255, np.uint8)
        out.append(np.concatenate([a, sep, b], axis=1))
    imageio.mimwrite(out_mp4, out, fps=20, codec="libx264", quality=8,
                     macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"])
    # representative frame ~70% through
    k = int(n * 0.7)
    imageio.imwrite(frames_png, out[k])
    print(f"{out_mp4} ({len(out)} frames) + {frames_png}")


if __name__ == "__main__":
    architecture()
    side_by_side_video(
        "output/videos_strong/spatial/indist/task04_baseline.mp4",
        "output/videos_strong/spatial/indist/task04_ph.mp4",
        str(OUT / "ph_wins_task4.mp4"), str(OUT / "ph_wins_task4_frame.png"))
    print("DONE assets")
