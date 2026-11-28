"""NutritionVerse serving API. Training helpers retain the registered 11/25 implementation."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from challenge_external.conformal import bounds
from challenge_external.frozen_1125 import nutritionverse_model as reference
from challenge_external.frozen_1125.nutritionverse_model import (
    DEVICE, KEYS, OUT, ROOT, PointPredictor, design, encoder, extract, fit, plate_view,
    points, read, scale, sha,
)

MODEL_SOURCE = "challenge_external/nutritionverse_model.py"
RELEASE = ROOT / "challenge_external/nutritionverse_extension/inference_release.json"


def validate_hashes(dependencies):
    """Check both the deployment and the byte-identical source used at experiment freeze."""
    release = read(RELEASE)
    reference_path = Path(reference.__file__).resolve()
    if sha(__file__) != release["serving_sha256"] or sha(reference_path) != release["reference_sha256"]:
        raise ValueError("NutritionVerse inference code differs from the released code")
    resolved = dict(dependencies)
    if resolved.get(MODEL_SOURCE) == release["reference_sha256"]:
        resolved[reference_path.relative_to(ROOT).as_posix()] = resolved.pop(MODEL_SOURCE)
    reference.validate_hashes(resolved)


class NutritionVerseSession:
    def __init__(self):
        from model_lora import CLIPLoRAMultiTask

        cal_bytes = (OUT / "calibration.json").read_bytes()
        freeze_bytes = (OUT / "freeze.json").read_bytes()
        self.cal = json.loads(cal_bytes)
        self.freeze = json.loads(freeze_bytes)
        if self.cal["freeze_sha256"] != hashlib.sha256(freeze_bytes).hexdigest():
            raise ValueError("Calibration does not match the frozen selection")
        validate_hashes(self.freeze["dependencies"])
        self.calibration_version = "nv90-" + hashlib.sha256(cal_bytes).hexdigest()[:12]
        self.model_version = "nv-ridge-" + self.cal["freeze_sha256"][:12]
        head_bytes = (OUT / "head.npz").read_bytes()
        if hashlib.sha256(head_bytes).hexdigest() != self.freeze["dependencies"]["challenge_external/nutritionverse_extension/head.npz"]:
            raise ValueError("Regression head does not match the frozen selection")
        with np.load(io.BytesIO(head_bytes)) as data:
            self.head = dict(data)

        ck = torch.load(ROOT / "checkpoints_lora_mt/best_model.pt", map_location="cpu", weights_only=False)
        state = ck["model_state"]
        model = CLIPLoRAMultiTask(num_ingredients=state["ingredient_head.weight"].shape[0],
                                  last_n=ck["lora_last_n"], rank=ck["rank"], alpha=ck["alpha"])
        model.load_state_dict(state, strict=True)
        self.vision = model.clip.to(DEVICE).eval()
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.normalize = transforms.Normalize([.4815, .4578, .4082], [.2686, .2613, .2758])

    def _vector(self, rgb):
        tensor = self.transform(rgb).unsqueeze(0).to(DEVICE)
        return self.vision(pixel_values=self.normalize(tensor)).pooler_output.cpu().numpy().astype(float)

    @torch.inference_mode()
    def predict_rgb(self, rgb):
        """Predict from an already decoded PIL RGB image, with the same resize and two forwards as training."""
        if rgb.mode != "RGB":
            raise ValueError("predict_rgb expects a PIL RGB image")
        crop, descriptor = plate_view(rgb)
        blocks = (self._vector(rgb), self._vector(crop), descriptor[None, :])
        p = points(blocks, self.head, self.freeze["selected"])[0]
        radius = scale(p, self.freeze["typical"], self.freeze["scale"]) * self.cal["q"]["adapted"]
        lo, hi = bounds(p, radius)
        return {"model_version": self.model_version, "calibration_version": self.calibration_version,
                "target_coverage": self.cal["target_coverage"], "coverage_scope": "per_field_marginal_nutritionverse_dishes",
                "fields": {k: {"point": float(p[i]), "lower": float(lo[i]), "upper": float(hi[i]),
                               "unit": "kcal" if k == "total_calories_kcal" else "g"}
                           for i, k in enumerate(KEYS)}}

    def predict(self, image):
        with Image.open(image) as im:
            return self.predict_rgb(im.convert("RGB"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Five nutritional intervals from a single RGB image")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    print(json.dumps(NutritionVerseSession().predict(args.image), ensure_ascii=False, indent=2))
