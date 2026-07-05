"""
make_diagrams.py — build all editable architecture diagrams for the report as SVG.
Run with any python that can import svgkit (no third-party deps).

Produces report/diagrams_svg/*.svg. A separate step converts these to EMF (for
native editing inside Word) and PNG (for preview).

Model diagrams (SmolVLA / ACT / pi0.5 / GR00T) share one template with:
  * an outer dashed boundary around the whole model,
  * an inner dashed boundary around the DeepONet action head,
  * the persistent-homology (PH) loss shown as a training-only box.
"""
import os
from svgkit import Canvas, BLUE, ORANGE, GREEN, GREY, PURPLE, YELLOW, RED

OUT = os.path.join(os.path.dirname(__file__), "diagrams_svg")
os.makedirs(OUT, exist_ok=True)


# ===========================================================================
# Shared template for the four model diagrams
# ===========================================================================
def model_diagram(fname, title, backbone_name, backbone_note, token_label,
                  chunk_note, baseline_head):
    c = Canvas(1080, 468, title)
    # outer boundary — the whole model
    c.region(22, 74, 1036, 330, "Whole model", color="#5b6b7a", label_fill="#39505f")
    # inputs -> backbone -> prefix tokens
    c.box(44, 196, 150, 86, "Inputs", "images + language + robot state", GREY)
    c.box(214, 184, 166, 112, backbone_name, backbone_note, BLUE)
    c.arrow(194, 239, 212, 239)
    c.text(430, 206, token_label, size=11, fill="#555")
    c.small_tokens(390, 222, 5, fill=BLUE)
    c.arrow(382, 239, 389, 234)
    # inner boundary — the DeepONet action head
    c.region(466, 152, 392, 168, "DeepONet action head", color="#7a52c0")
    c.box(482, 200, 120, 76, "Cross-attention pooler", "sees the whole input", PURPLE)
    c.box(616, 200, 116, 76, "branch × trunk", "operator", ORANGE)
    c.box(746, 200, 96, 76, "output MLP", "", ORANGE)
    c.arrow(471, 239, 480, 239)      # tokens -> pooler
    c.arrow(602, 239, 614, 239)      # pooler -> branch
    c.arrow(732, 239, 744, 239)      # branch -> output
    # action chunk
    c.box(886, 200, 152, 76, "Action chunk", chunk_note, YELLOW)
    c.arrow(842, 239, 884, 239)      # output -> chunk
    # persistent-homology loss (training only)
    c.box(598, 342, 120, 48, "Expert chunk", "", GREY)
    c.box(758, 338, 196, 58, "MSE + λ·PH loss", "(training only)", GREEN)
    c.arrow(962, 276, 902, 336)      # predicted chunk -> loss
    c.arrow(718, 366, 756, 366)      # expert chunk -> loss
    c.text(540, 436,
           f"Drop-in replacement for the native {baseline_head} head — only the action head "
           f"changes between the three variants (baseline / DeepONet / DeepONet + PH).",
           size=11, fill="#333")
    c.save(os.path.join(OUT, fname))


# ===========================================================================
# 1. What a VLA is (big picture, closed loop)
# ===========================================================================
def fig_vla_overview():
    c = Canvas(980, 430, "Figure 1.  A vision-language-action (VLA) model, at a glance")
    c.box(30, 95, 150, 48, "Camera images", "main + wrist", GREY)
    c.box(30, 165, 150, 48, "Language", "\"put the bowl ...\"", GREY)
    c.box(30, 235, 150, 48, "Robot state", "arm + gripper", GREY)
    c.box(250, 120, 175, 150, "VLM backbone", "vision + language\n(usually frozen)", BLUE)
    c.arrow(180, 119, 248, 150); c.arrow(180, 189, 248, 195); c.arrow(180, 259, 248, 245)
    c.text(470, 150, "prefix tokens", size=11, fill="#555")
    c.small_tokens(452, 160, 6, fill=BLUE)
    c.arrow(425, 195, 452, 178)
    c.box(560, 130, 170, 130, "Action head", "the part we study", ORANGE)
    c.arrow(540, 178, 558, 190)
    c.box(770, 155, 175, 90, "Action chunk", "a(τ), τ in [0,1]", YELLOW)
    c.arrow(730, 195, 768, 198)
    c.box(770, 300, 175, 60, "Robot executes", "a few steps", GREEN)
    c.arrow(857, 245, 857, 298)
    c.arrow(770, 330, 105, 288, "execute a few steps, then look again (receding horizon)",
            color="#3a9457", curve=(430, 405), label_dy=16)
    c.save(os.path.join(OUT, "fig01_vla_overview.svg"))


# ===========================================================================
# 2. SmolVLA (model diagram)
# ===========================================================================
def fig_smolvla():
    model_diagram("fig02_smolvla.svg",
                  "Figure 2.  SmolVLA with the DeepONet action head",
                  "SmolVLM2 backbone", "vision + language\n~350M",
                  "prefix tokens", "50 steps × 7-DoF", "flow-matching")


# ===========================================================================
# 3. Flow-matching head detail
# ===========================================================================
def fig_flow():
    c = Canvas(980, 420, "Figure 3.  How the flow-matching action head works")
    c.box(40, 90, 160, 55, "Prefix tokens", "from backbone", BLUE)
    c.box(40, 175, 160, 55, "Noisy action x_t", "x_t = t·noise + (1-t)·action", GREY)
    c.box(40, 260, 160, 55, "Time t in [0,1]", "", GREY)
    c.box(300, 150, 220, 130, "Transformer\n(cross-attention)", "predicts a velocity field v_t",
          ORANGE, title_dy=48)
    c.arrow(200, 117, 298, 175); c.arrow(200, 202, 298, 205); c.arrow(200, 287, 298, 245)
    c.box(600, 165, 180, 90, "Velocity v_t", "denoising direction", PURPLE)
    c.arrow(520, 215, 598, 210)
    c.box(600, 300, 340, 60, "Clean action = x_t − t·v_t   (one formula, one pass)", "", YELLOW)
    c.arrow(690, 255, 690, 298)
    c.box(820, 175, 130, 70, "Training loss", "MSE(v_t, target)", GREEN)
    c.arrow(780, 210, 818, 210)
    c.save(os.path.join(OUT, "fig03_flowmatching.svg"))


# ===========================================================================
# 4. Persistent homology idea (conceptual)
# ===========================================================================
def fig_ph_idea():
    c = Canvas(980, 430, "Figure 4.  The persistent-homology (PH) idea: compare the SHAPE of trajectories")
    c.text(200, 74, "Predicted action chunk", size=13, weight="bold")
    c.text(760, 74, "Expert action chunk", size=13, weight="bold")

    def cloud(cx, cy, seedpts, color):
        f, s = color
        for (dx, dy) in seedpts:
            c.els.append(f'<circle cx="{cx+dx}" cy="{cy+dy}" r="5" fill="{f}" stroke="{s}" stroke-width="1.5"/>')
    pts = [(-60, 10), (-30, -25), (0, 5), (25, -20), (55, 15), (15, 40), (-20, 35), (40, -40)]
    cloud(200, 170, pts, PURPLE)
    cloud(760, 170, [(x+6, y-4) for (x, y) in pts], GREEN)
    c.text(200, 250, "the 50 timesteps as a cloud", size=10, fill="#555")
    c.text(200, 265, "of points in action space", size=10, fill="#555")
    c.text(760, 258, "same, for the expert", size=10, fill="#555")
    c.box(350, 300, 280, 52, "top-k pairwise distances", "the \"topological fingerprint\"", GREY)
    c.arrow(240, 205, 380, 298, "measure distances", color="#7a52c0")
    c.arrow(730, 205, 600, 298, "measure distances", color="#3a9457")
    c.box(350, 372, 280, 42, "PH loss = match the two fingerprints", "", GREEN)
    c.arrow(490, 352, 490, 370)
    c.text(490, 116, "Idea: two motions can hit different points yet share the same overall shape.",
           size=11, fill="#333")
    c.save(os.path.join(OUT, "fig04_ph_idea.svg"))


# ===========================================================================
# 5. Where the PH loss plugs in
# ===========================================================================
def fig_ph_plug():
    c = Canvas(940, 320, "Figure 5.  PH is an extra training-time loss (inference is untouched)")
    c.box(60, 120, 180, 70, "Action head", "flow or DeepONet", ORANGE)
    c.box(320, 70, 180, 60, "Predicted chunk", "", YELLOW)
    c.box(320, 170, 180, 60, "Expert chunk", "ground truth", GREY)
    c.arrow(240, 145, 318, 100)
    c.box(600, 60, 150, 60, "MSE loss", "point-wise", GREEN)
    c.box(600, 150, 150, 60, "PH loss", "shape / topology", GREEN)
    c.arrow(500, 100, 598, 90); c.arrow(500, 200, 598, 180); c.arrow(500, 100, 598, 175, color="#3a9457")
    c.box(790, 105, 130, 70, "Total loss", "MSE + λ·PH", PURPLE)
    c.arrow(750, 90, 788, 120); c.arrow(750, 180, 788, 155)
    c.text(470, 288, "λ is a small weight (e.g. 0.02). At test time only the action head runs — no losses, no slowdown.",
           size=11, fill="#333")
    c.save(os.path.join(OUT, "fig05_ph_plug.svg"))


# ===========================================================================
# 6. DeepONet head (the core), with an operator boundary
# ===========================================================================
def fig_deeponet():
    c = Canvas(1010, 470, "Figure 6.  Inside the DeepONet operator action head")
    c.region(214, 84, 660, 318, "DeepONet action head", color="#7a52c0")
    c.box(34, 118, 150, 60, "Prefix tokens", "(B, N, D)", BLUE)
    c.box(240, 96, 200, 104, "Cross-attention pooler", "K=8 queries read all tokens\n(keeps spatial detail)",
          PURPLE, title_dy=40)
    c.arrow(184, 148, 238, 148)
    c.box(500, 108, 150, 80, "Branch MLP", "→ coefficients c", ORANGE)
    c.arrow(440, 148, 498, 146)
    c.text(575, 208, "c in R^p  (p=256)", size=10, fill="#555")

    c.box(34, 300, 150, 60, "Time τ in [0,1]", "50 steps", GREY)
    c.box(240, 286, 200, 84, "Fourier features", "τ → [τ, sin, cos ...]", GREY)
    c.arrow(184, 330, 238, 330)
    c.box(500, 290, 150, 80, "Trunk MLP", "→ basis φ(τ)", ORANGE)
    c.arrow(440, 330, 498, 330)
    c.text(575, 388, "φ(τ) in R^p", size=10, fill="#555")

    c.box(712, 200, 130, 80, "Merge", "c × φ(τ)\nelement-wise", GREEN, title_dy=36)
    c.arrow(650, 148, 722, 202, color="#c2772e")
    c.arrow(650, 330, 722, 278, color="#c2772e")
    c.box(884, 205, 100, 70, "Output MLP", "", ORANGE)
    c.arrow(842, 240, 882, 240)
    c.box(884, 320, 116, 70, "Action chunk", "a(τ)  (B,T,A)", YELLOW)
    c.arrow(934, 275, 934, 318)
    c.text(505, 448,
           "One forward pass — no denoising loop.  a(τ) = OutMLP( branch(obs) × trunk(τ) ).  "
           "~10-11M params. Trained with MSE (+ optional PH loss).", size=11, fill="#333")
    c.save(os.path.join(OUT, "fig06_deeponet.svg"))


# ===========================================================================
# 7. DeepONet + PH (loss view)
# ===========================================================================
def fig_deeponet_ph():
    c = Canvas(960, 340, "Figure 7.  DeepONet + PH variant (same head, extra training loss)")
    c.box(50, 120, 180, 90, "DeepONet head", "operator, one pass", ORANGE)
    c.box(320, 115, 170, 60, "Predicted chunk", "", YELLOW)
    c.arrow(230, 165, 318, 150)
    c.box(320, 210, 170, 60, "Expert chunk", "", GREY)
    c.box(580, 95, 150, 60, "MSE loss", "", GREEN)
    c.box(580, 185, 150, 60, "PH loss", "optional, λ", GREEN)
    c.arrow(490, 145, 578, 125); c.arrow(490, 240, 578, 215); c.arrow(490, 145, 578, 205, color="#3a9457")
    c.box(770, 135, 140, 70, "Total loss", "MSE + λ·PH", PURPLE)
    c.arrow(720, 125, 768, 155); c.arrow(720, 215, 768, 180)
    c.text(480, 312, "Same operator head; PH just adds a topological shape term during training.",
           size=11, fill="#333")
    c.save(os.path.join(OUT, "fig07_deeponet_ph.svg"))


# ===========================================================================
# 8-10. ACT / pi0.5 / GR00T (model diagrams)
# ===========================================================================
def fig_act():
    model_diagram("fig08_act.svg",
                  "Figure 8.  ACT with the DeepONet action head",
                  "ResNet-18 + encoder", "vision + language\n~15M",
                  "memory tokens", "100 steps × 7-DoF", "transformer-decoder")


def fig_pi05():
    model_diagram("fig09_pi05.svg",
                  "Figure 9.  pi0.5 with the DeepONet action head (SOTA comparison)",
                  "PaliGemma backbone", "vision + language\n~3.3B, frozen",
                  "prefix tokens", "50 × 32 (7 used)", "flow-matching")


def fig_groot():
    model_diagram("fig10_groot.svg",
                  "Figure 10.  GR00T N1.6 with the DeepONet action head (SOTA comparison)",
                  "Eagle VLM backbone", "vision + language\n~2B, frozen",
                  "VL tokens", "16 × 29 (7 used)", "32-layer diffusion")


# ===========================================================================
# 11. Protocol
# ===========================================================================
def fig_protocol():
    c = Canvas(1000, 340, "Figure 11.  The training and evaluation protocol (same for every model)")
    c.box(40, 110, 190, 90, "Pre-train", "40 tasks together\n(15K steps)", BLUE, title_dy=34)
    c.box(300, 110, 190, 90, "Fine-tune", "per suite, 10 tasks\n(15K steps)", ORANGE, title_dy=34)
    c.arrow(230, 155, 298, 155, "start from\npre-trained", label_dy=-14)
    c.box(560, 60, 200, 75, "In-distribution eval", "same tasks,\nmany episodes", GREEN, title_dy=30)
    c.box(560, 165, 200, 75, "LIBERO-Plus eval", "7 OOD perturbation\ncategories", RED, title_dy=30)
    c.arrow(490, 145, 558, 100); c.arrow(490, 165, 558, 195)
    c.box(820, 110, 150, 90, "Scores", "accuracy +\nrobustness", YELLOW, title_dy=34)
    c.arrow(760, 97, 818, 140); c.arrow(760, 202, 818, 165)
    c.text(500, 300, "Repeated for each variant and (where noted) several random seeds.", size=11, fill="#333")
    c.save(os.path.join(OUT, "fig11_protocol.svg"))


# ===========================================================================
# 12. LIBERO-Plus perturbation categories
# ===========================================================================
def fig_libero_plus():
    c = Canvas(1010, 340, "Figure 12.  LIBERO-Plus: seven ways the world is changed to test robustness")
    cats = ["Camera\nviewpoints", "Lighting\nconditions", "Sensor\nnoise", "Background\ntextures",
            "Object\nlayout", "Robot initial\nstate", "Language\nwording"]
    x = 34
    for i, name in enumerate(cats):
        col = RED if i % 2 == 0 else ORANGE
        c.box(x, 120, 126, 92, name, "", col, title_dy=44)
        x += 139
    c.text(505, 74, "The model is trained on the normal world, then tested on each perturbed version it has never seen.",
           size=12, fill="#333")
    c.text(505, 262, "Robustness score = average success across these seven categories.", size=11, fill="#555")
    c.save(os.path.join(OUT, "fig12_libero_plus.svg"))


if __name__ == "__main__":
    for fn in [fig_vla_overview, fig_smolvla, fig_flow, fig_ph_idea, fig_ph_plug,
               fig_deeponet, fig_deeponet_ph, fig_act, fig_pi05, fig_groot,
               fig_protocol, fig_libero_plus]:
        fn()
    print("wrote", len([f for f in os.listdir(OUT) if f.endswith('.svg')]), "SVGs to", OUT)
