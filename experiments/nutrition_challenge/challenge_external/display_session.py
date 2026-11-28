"""One frozen inference session for repeated calls from the FoodLens display."""
import argparse
import json

from PIL import Image
from challenge_external.conformal import bounds
from challenge_external.runtime import KEYS, MODELS, RESULTS, ROOT, PointPredictor, sha


class IntervalSession:
    """Load and verify the default release once. Recreate after changing versions."""

    def __init__(self):
        self._artifact = json.loads((RESULTS / "calibration.json").read_text())
        for relative, expected in self._artifact["inference_dependencies"].items():
            if sha(ROOT / relative) != expected:
                raise ValueError(f"Calibration incompatible with {relative}")
        self._predictor = PointPredictor()

    def predict_all(self, image_path):
        with Image.open(image_path) as image:
            rgb = self._predictor.tf(image.convert("RGB")).unsqueeze(0)
        predictions = self._predictor.predict_tensors(rgb)
        output = {}
        for model_id in MODELS:
            model = self._artifact["models"][model_id]
            point = predictions[model_id][0]
            lower, upper = bounds(point, model["q"])
            output[model_id] = {
                "model_id": model_id,
                "model_version": model["model_version"],
                "calibration_version": self._artifact["version"],
                "target_coverage": self._artifact["target_coverage"],
                "coverage_scope": "per_field_marginal",
                "fields": {
                    key: {"point": float(point[i]), "lower": float(lower[i]),
                          "upper": float(upper[i]), "unit": "kcal" if i == 1 else "g"}
                    for i, key in enumerate(KEYS)
                },
                "limitations": "Calibrated on OmniFood8K official train; batch independence and coverage on other domains are not established.",
            }
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all three frozen baselines on one RGB image")
    parser.add_argument("image")
    args = parser.parse_args()
    print(json.dumps(IntervalSession().predict_all(args.image), ensure_ascii=False, indent=2))
