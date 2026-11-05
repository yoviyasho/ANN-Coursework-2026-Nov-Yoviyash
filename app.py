"""
ForestNet driver classifier — web application (Question 4 / 5.b)

Frontend + backend in one Gradio app. Every prediction is computed by the trained ViT at request time:
  upload image -> input checks -> preprocessing (as in training) -> ViT -> temperature-scaled scores
  -> recommended action -> Grad-CAM explanation.
Files needed next to this script: config.json, model.pt (and optionally examples/*.png).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm

HERE = Path(__file__).resolve().parent
CFG = json.load(open(HERE / "config.json"))
CLASSES = CFG["classes"]                         # alphabetical order used in training
ACTIONS = CFG["actions"]
T = float(CFG["temperature"])                    # temperature fitted on the validation set
LOW_CONF = float(CFG["low_confidence_threshold"])
IMG = int(CFG["img_size"])
MEAN = np.array(CFG["mean"], dtype=np.float32)   # training-set channel mean (0-1 scale)
STD = np.array(CFG["std"], dtype=np.float32)
OOD = CFG["ood_ranges"]
torch.set_num_threads(max(1, torch.get_num_threads()))


# ----------------------------------------------------------------------------- model (same as training)
class ViTClassifier(nn.Module):
    def __init__(self, depth=12, hidden=256, drop=0.2):
        super().__init__()
        self.backbone = timm.create_model(CFG["timm_name"], pretrained=False, num_classes=0)
        self.backbone.blocks = self.backbone.blocks[:depth]
        d = self.backbone.num_features
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(drop), nn.Linear(2 * d, hidden),
                                  nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, len(CLASSES)))

    def forward(self, x):
        t = self.backbone.forward_features(x)
        return self.head(torch.cat([t[:, 0], t[:, 1:].mean(1)], dim=1))


model = ViTClassifier(CFG["depth"], CFG["hidden"], CFG["drop"])
model.load_state_dict(torch.load(HERE / "model.pt", map_location="cpu"))
model.eval()


# ----------------------------------------------------------------------------- Grad-CAM adapted for ViT
class ViTGradCAM:
    """Patch tokens entering the last block's attention are reshaped to a 14 x 14 grid ([CLS] dropped)."""
    def __init__(self, m):
        self.acts, self.grads = None, None
        m.backbone.blocks[-1].norm1.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, target):
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(x)
            logits[0, target].backward()
        a, g = self.acts[0, 1:, :].detach(), self.grads[0, 1:, :]
        grid = int(round(a.shape[0] ** 0.5))
        cam = F.relu((g.mean(0, keepdim=True) * a).sum(-1)).view(grid, grid)
        return (cam / (cam.max() + 1e-8)).numpy()


gradcam = ViTGradCAM(model)


# ----------------------------------------------------------------------------- input handling
def check_and_prepare(img):
    """Returns (rgb uint8 array or None, error message or None, list of warnings, out-of-distribution flag)."""
    if img is None:
        return None, "Please upload an image.", [], False
    img = np.asarray(img)
    warnings = []
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
        warnings.append("The image is greyscale; the model was trained on colour (RGB) composites.")
    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        return None, "Unsupported image format. Please upload an RGB satellite image (PNG or JPG).", [], False
    img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    if min(h, w) < 64:
        return None, f"The image is too small ({w} x {h} px). Please upload at least 64 x 64 px (ideally 332 x 332).", [], False
    if max(h, w) / min(h, w) > 1.5:
        warnings.append("The image is far from square; it will be stretched to a square, which may affect the result.")
    stats = {"mean_brightness": float(img.mean()), "contrast": float(img.std()),
             "green_ratio": float(img[..., 1].mean() / (img.mean() + 1e-6))}
    unusual = [k for k, v in stats.items() if not (OOD[k][0] <= v <= OOD[k][1])]
    if unusual:
        warnings.append("This image looks very different from the satellite composites the model was trained on "
                        f"({', '.join(k.replace('_', ' ') for k in unusual)} outside the training range).")
    return img, None, warnings, bool(unusual)


def preprocess(img_u8):
    x = np.asarray(Image.fromarray(img_u8).resize((IMG, IMG), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1)[None].float(), \
        np.asarray(Image.fromarray(img_u8).resize((IMG, IMG), Image.BILINEAR))


def stretch(img_u8):
    """Contrast enhancement for display only (ForestNet composites are very dark)."""
    lo, hi = np.percentile(img_u8, [2, 98])
    return np.clip((img_u8.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)


def overlay(img_u8, cam, alpha=0.45):
    base = stretch(img_u8)
    heat = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(base.shape[1::-1], Image.BILINEAR)) / 255.0
    colour = cm.jet(heat)[..., :3]
    return ((1 - alpha) * base + alpha * colour * 1.0).clip(0, 1)


# ----------------------------------------------------------------------------- prediction (called per request)
def predict(img):
    rgb, err, warns, ood = check_and_prepare(img)
    if err:
        return ({}, f"### ⚠️ {err}", None, None, "")
    x, small = preprocess(rgb)
    with torch.no_grad():
        logits = model(x)[0]
    probs = F.softmax(logits / T, dim=0).numpy()            # calibrated (temperature-scaled) scores
    k = int(probs.argmax())
    cam = gradcam(x, k)
    cls = CLASSES[k]
    scores = {CFG["display_names"][c]: float(p) for c, p in zip(CLASSES, probs)}
    caption = ("**How to read the explanation:** red/yellow areas had the strongest influence on this prediction; "
               "blue areas had little influence. This shows what the model *looked at*, not proof of what caused "
               "the forest loss. Check whether the highlighted areas contain relevant evidence (e.g. plantation "
               "grids, mining scars, small fields) before acting.")

    if ood:   # unusual input: withhold the prediction and the action instead of showing a misleading score
        lines = ["## ⚠️ This image does not look like a ForestNet satellite composite",
                 *[f"**{w}**" for w in warns],
                 "**No driver or action is recommended**, because the model was trained only on Landsat 8 RGB "
                 "composites of forest-loss events in Indonesia. Please upload a 332 × 332 px composite centred on a "
                 "forest-loss event.",
                 f"For transparency: the model's output would have been *{CFG['display_names'][cls]}* with score "
                 f"{probs[k]:.2f}, but **scores are not meaningful for this kind of image** — the model always picks "
                 "one of its four classes, often with a high score, even for unrelated images."]
        return {}, "\n\n".join(lines), stretch(small), overlay(small, cam), caption

    lines = [f"## Predicted driver: **{CFG['display_names'][cls]}**",
             f"**Recommended action:** {ACTIONS[cls]}",
             f"**Calibrated score:** {probs[k]:.2f}"]
    if probs[k] < LOW_CONF:
        lines.append(f"### ⚠️ Low certainty (score below {LOW_CONF:.2f}) — refer this event for **manual review** "
                     "before acting.")
    second = int(np.argsort(probs)[-2])
    if probs[k] - probs[second] < 0.2:
        lines.append(f"The model also considered **{CFG['display_names'][CLASSES[second]]}** "
                     f"(score {probs[second]:.2f}).")
    for w in warns:
        lines.append(f"> ⚠️ {w}")
    return scores, "\n\n".join(lines), stretch(small), overlay(small, cam), caption


# ----------------------------------------------------------------------------- user interface
ABOUT_SCORES = f"""
**What the score means.** The raw output of the network was found to be **overconfident** (on validation data,
24 of 88 errors had a raw probability above 0.99). The scores shown here are **temperature-scaled**
(T = {T:.2f}, fitted on the validation set) so that they better reflect how often predictions are correct.
A score of 0.80 means that, on validation data, predictions with similar scores were correct roughly 80% of the
time. It is **not a guarantee** for an individual image. Scores below {LOW_CONF:.2f} are flagged for manual review.
"""

LIMITATIONS = f"""
- Trained on **{CFG['n_train']} Landsat 8 composites of forest-loss events in Indonesia (2001–2016)**. Images from other
  countries, other satellites, other years or other processing may give unreliable results.
- Only **four classes** (Plantation, Smallholder agriculture, Grassland/shrubland, Other). It cannot recognise any other
  land use.
- Uses the **image only** (no elevation, climate or ownership data).
- Validation macro-F1 ≈ **{CFG['val_macro_f1']:.2f}** — about 1 in 5 events is misclassified. It is weakest on
  **Other** (especially regrowing secondary forest, often predicted as Smallholder), on **Grassland**, on very small
  loss areas, and in the eastern islands. About 1 in 10 plantations is predicted as Smallholder.
- The input should be a **332 × 332 px (≈ 5 km) RGB composite centred on the loss event**, like the ForestNet data.
- The explanation is approximate (each grid cell ≈ 360 m) and **does not prove the cause** of a prediction.
- This is a **decision-support tool**: it does not replace field verification or legal evidence.
"""

with gr.Blocks(title="ForestNet deforestation driver classifier") as demo:
    gr.Markdown("# 🌳 Deforestation driver classifier (ForestNet, Indonesia)\n"
                "Upload a Landsat 8 RGB composite of a forest-loss event. The model predicts the most likely "
                "**direct driver**, the **recommended action**, a **calibrated score**, and shows **which parts of "
                "the image influenced the prediction**.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="Satellite image (RGB composite)", image_mode="RGB")
            btn = gr.Button("Classify", variant="primary")
            ex_dir = HERE / "examples"
            if ex_dir.exists() and any(ex_dir.glob("*.png")):
                gr.Examples(sorted(str(p) for p in ex_dir.glob("*.png")), inputs=inp, label="Example images")
        with gr.Column(scale=1):
            decision = gr.Markdown()
            scores = gr.Label(num_top_classes=4, label="Calibrated scores")
    with gr.Row():
        orig = gr.Image(label="Image (contrast enhanced for viewing)")
        expl = gr.Image(label="Explanation (Grad-CAM)")
    caption = gr.Markdown()
    with gr.Accordion("What does the score mean?", open=False):
        gr.Markdown(ABOUT_SCORES)
    with gr.Accordion("Limitations of this model", open=True):
        gr.Markdown(LIMITATIONS)
    btn.click(predict, inputs=inp, outputs=[scores, decision, orig, expl, caption])

if __name__ == "__main__":
    demo.launch()
