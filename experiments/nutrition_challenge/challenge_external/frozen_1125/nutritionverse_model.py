"""Single-RGB plate adaptation and versioned intervals for NutritionVerse-Real."""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from challenge_external.conformal import bounds
from challenge_external.runtime import DEVICE, KEYS, ROOT, PointPredictor, sha

OUT = ROOT / "challenge_external/nutritionverse_extension"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


@lru_cache(maxsize=1)
def encoder():
    return PointPredictor()


def plate_view(rgb):
    # Estimate the photographed plate extent from pixels only. Use the full frame
    # as a fallback when the lighting gives too little foreground evidence.
    thumbnail = rgb.resize((96, 96), Image.Resampling.BILINEAR)
    arr = np.asarray(thumbnail, dtype=float) / 255
    hi, lo = arr.max(2), arr.min(2)
    chroma = hi - lo
    foreground = (lo > .56) | ((chroma > .17) & (hi > .35))
    foreground[:4] = foreground[-4:] = False
    foreground[:, :4] = foreground[:, -4:] = False
    yy, xx = np.where(foreground)
    crop = rgb
    if len(xx) >= 150:
        x0, x1 = np.quantile(xx, [.02, .98])
        y0, y1 = np.quantile(yy, [.02, .98])
        dx, dy = max(4., .09*(x1-x0)), max(4., .09*(y1-y0))
        w, h = rgb.size
        box = (max(0, int((x0-dx)*w/96)), max(0, int((y0-dy)*h/96)),
               min(w, int(np.ceil((x1+dx+1)*w/96))), min(h, int(np.ceil((y1+dy+1)*h/96))))
        if box[2] > box[0] and box[3] > box[1]:
            crop = rgb.crop(box)
    pixels = np.asarray(crop.resize((64, 64), Image.Resampling.BILINEAR), dtype=float) / 255
    high, low = pixels.max(2), pixels.min(2)
    color = high - low
    descriptors = np.array([(color > t).mean() for t in (.12, .22, .35, .5)] +
                           [(low > .65).mean(), (high < .3).mean(), foreground.mean(),
                            *pixels.mean(axis=(0, 1)), *pixels.std(axis=(0, 1))])
    return crop, descriptors


@torch.inference_mode()
def extract(paths):
    net = encoder()
    full_vectors, plate_vectors, sizes, resnet, clip = [], [], [], [], []
    for start in range(0, len(paths), 8):
        full, plates = [], []
        for path in paths[start:start+8]:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                crop, descriptor = plate_view(rgb)
                full.append(net.tf(rgb))
                plates.append(net.tf(crop))
                sizes.append(descriptor)
        x = torch.stack(full).to(DEVICE)
        full_feat = net.clip.clip(pixel_values=net.cnorm(x)).pooler_output
        crop_feat = net.clip.clip(pixel_values=net.cnorm(torch.stack(plates).to(DEVICE))).pooler_output
        raw = (net.clip.norm_head(full_feat).cpu()*net.cs+net.cm).numpy()
        clip.extend([[round(max(0, float(v)), 1) for v in row] for row in raw])
        resnet.extend(np.maximum(0, net.resnet(net.rnorm(x)).cpu().numpy()*net.std+net.mean))
        full_vectors.extend(full_feat.cpu().numpy())
        plate_vectors.extend(crop_feat.cpu().numpy())
    blocks = tuple(np.asarray(block, dtype=float) for block in (full_vectors, plate_vectors, sizes))
    return blocks, {"resnet18": np.asarray(resnet), "clip_lora": np.asarray(clip)}


def design(blocks, mode):
    # Unit-length embeddings preserve angular information. Normalize each block
    # separately so the size features can contribute on a comparable scale.
    full, plate, sizes = blocks
    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    if mode == "full":
        return unit(full)
    features = [unit(full), unit(plate)]
    if mode == "plate_size":
        features.append(sizes / np.sqrt(sizes.shape[1]))
    return np.concatenate(features, axis=1)


def fit(blocks, targets, config):
    x = design(blocks, config["view"])
    xc = x - x.mean(0)
    gram = xc @ xc.T
    penalty = max(1e-10, config["ridge"] * np.trace(gram)/len(x))
    y = np.sqrt(targets) if config["transform"] == "sqrt" else targets
    coef = xc.T @ np.linalg.solve(gram + penalty*np.eye(len(x)), y-y.mean(0))
    return {"center": x.mean(0), "coef": coef, "intercept": y.mean(0)}


def points(blocks, head, config):
    x = design(blocks, config["view"])
    pred = np.maximum(0, (x-head["center"]) @ head["coef"] + head["intercept"])
    return np.square(pred) if config["transform"] == "sqrt" else pred


def scale(point, typical, method):
    if method == "absolute":
        return np.ones_like(point)
    if method == "portion_scaled":
        return np.sqrt(.25 + point / np.maximum(typical, 1e-8))
    raise ValueError(f"Unknown interval scaling: {method}")


def validate_hashes(dependencies):
    for relative, digest in dependencies.items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"Frozen file changed: {relative}")


class NutritionVerseSession:
    def __init__(self):
        self.freeze = read(OUT / "freeze.json")
        self.cal = read(OUT / "calibration.json")
        if self.cal["freeze_sha256"] != sha(OUT / "freeze.json"):
            raise ValueError("Calibration does not match the frozen selection")
        validate_hashes(self.freeze["dependencies"])
        with np.load(OUT / "head.npz") as data:
            self.head = dict(data)
        encoder()

    def predict(self, image):
        blocks, _ = extract([image])
        p = points(blocks, self.head, self.freeze["selected"])[0]
        radius = scale(p, self.freeze["typical"], self.freeze["scale"]) * self.cal["q"]["adapted"]
        lo, hi = bounds(p, radius)
        return {"model_version": "nv-ridge-"+self.cal["freeze_sha256"][:12],
                "calibration_version": "nv90-"+sha(OUT/"calibration.json")[:12],
                "target_coverage": .90, "coverage_scope": "per_field_marginal_nutritionverse_dishes",
                "fields": {k: {"point": float(p[i]), "lower": float(lo[i]), "upper": float(hi[i]),
                               "unit": "kcal" if k == "total_calories_kcal" else "g"}
                           for i, k in enumerate(KEYS)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Five nutritional intervals from a single RGB image")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    print(json.dumps(NutritionVerseSession().predict(args.image), ensure_ascii=False, indent=2))
