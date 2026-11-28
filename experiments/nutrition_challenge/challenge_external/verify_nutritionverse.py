"""Verify reported NutritionVerse results offline, including a real single-image call."""
import json
from math import ceil
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from challenge_external import nutritionverse_model
from challenge_external.nutritionverse_model import KEYS, OUT, ROOT, NutritionVerseSession, read, sha

DATA = ROOT / "challenge_external/data_supplement_v1/nutritionverse"


def verify_calibration_snapshot(image):
    # Change only temporary artifacts. A running display must keep the thresholds
    # and the version from the same bytes even across an on-disk replacement.
    with TemporaryDirectory() as directory:
        tmp = Path(directory)
        for name in ("freeze.json", "head.npz", "calibration.json"):
            copyfile(OUT / name, tmp / name)
        cal_path = tmp / "calibration.json"
        first_version = "nv90-" + sha(cal_path)[:12]
        with patch.object(nutritionverse_model, "OUT", tmp):
            session = NutritionVerseSession()
            original = session.predict(image)
            updated = read(cal_path)
            updated["q"]["adapted"] = [v * 1.25 for v in updated["q"]["adapted"]]
            candidate = tmp / "replacement.json"
            candidate.write_text(json.dumps(updated), encoding="utf8")
            candidate.replace(cal_path)
            with Image.open(image) as im:
                assert session.predict_rgb(im.convert("RGB")) == original
            assert original["calibration_version"] == first_version
            del session
            replacement = NutritionVerseSession().predict(image)
            assert replacement["calibration_version"] == "nv90-" + sha(cal_path)[:12] != first_version
            for k in KEYS:
                before = original["fields"][k]
                after = replacement["fields"][k]
                assert before["point"] == after["point"] and after["upper"] > before["upper"]
    return original


def main():
    registration = read(OUT / "protocol.json")
    manifests = {}
    for role, spec in registration["roles"].items():
        file = DATA / (role + ".jsonl")
        assert sha(file) == spec["sha256"]
        rr = [json.loads(line) for line in file.read_text(encoding="utf8").splitlines()]
        assert len(rr) == spec["n"] and all(r["role"] == role for r in rr)
        manifests[role] = rr
    for key in ("sample_id", "object_id", "image_sha256"):
        values = [r[key] for rr in manifests.values() for r in rr]
        assert len(set(values)) == len(values) == 225
    for rr in manifests.values():
        for r in rr:
            assert sha(DATA / r["image"]) == r["image_sha256"]

    fit, cal = read(OUT / "freeze.json"), read(OUT / "calibration.json")
    sets = {}
    for role, name in (("calibration_reserved", "calibration_predictions.json"),
                       ("test_reserved", "evaluation_predictions.json")):
        data = read(OUT / name)
        assert data["ids"] == [r["sample_id"] for r in manifests[role]]
        assert data["targets"] == [[r["target"][k] for k in KEYS] for r in manifests[role]]
        sets[role] = data
    c, e = sets["calibration_reserved"], sets["test_reserved"]
    assert sha(OUT / "freeze.json") == cal["freeze_sha256"]
    assert sha(OUT / "calibration_predictions.json") == cal["calibration_predictions_sha256"]
    assert sha(OUT / "calibration.json") == e["calibration_sha256"]
    assert set(c["points"]) == set(e["points"]) == set(cal["q"]) == set(registration["model_order"])
    rank = ceil((len(c["ids"])+1) * .9)
    assert rank == cal["rank"] == 55
    table = read(OUT / "metrics.json")
    misses = read(OUT / "undercoverage.json")
    targets = np.asarray(e["targets"])
    cal_targets = np.asarray(c["targets"])
    expected_misses = {}
    for model in registration["model_order"]:
        cp, p = np.array(c["points"][model]), np.array(e["points"][model])
        adaptive = model == "adapted" and fit["scale"] == "portion_scaled"
        cs = np.sqrt(.25 + cp/np.asarray(fit["typical"])) if adaptive else 1.
        es = np.sqrt(.25 + p/np.asarray(fit["typical"])) if adaptive else 1.
        q = np.sort(np.abs(cp-cal_targets)/cs, axis=0)[rank-1]
        np.testing.assert_allclose(q, cal["q"][model])
        lower, upper = np.maximum(0, p - es*q), p + es*q
        hit = (targets >= lower) & (targets <= upper)
        for j, field in enumerate(KEYS):
            row, = (r for r in table if r["model"] == model and r["field"] == field)
            err = p[:, j] - targets[:, j]
            width = upper[:, j] - lower[:, j]
            score = width + 20*(np.maximum(lower[:,j]-targets[:,j],0)+np.maximum(targets[:,j]-upper[:,j],0))
            np.testing.assert_allclose([row["mae"],row["mse"],row["hits"],row["coverage"],row["mean_width"],row["interval_score"]],
                [abs(err).mean(),np.square(err).mean(),hit[:,j].sum(),hit[:,j].mean(),width.mean(),score.mean()])
            for i in np.flatnonzero(~hit[:,j]):
                expected_misses[(model,e["ids"][i],field)] = [p[i,j], targets[i,j], lower[i,j], upper[i,j]]
    assert len(table) == 20 and len(misses) == len(expected_misses)
    for r in misses:
        np.testing.assert_allclose([r[k] for k in ("point","truth","lower","upper")],
                                   expected_misses.pop((r["model"],r["dish"],r["field"])))
    assert not expected_misses
    first = manifests["test_reserved"][0]
    out = verify_calibration_snapshot(DATA / first["image"])
    assert out["target_coverage"] == .9 and set(out["fields"]) == set(KEYS)
    for j,k in enumerate(KEYS):
        f = out["fields"][k]
        p = e["points"]["adapted"][0][j]
        factor = np.sqrt(.25+p/fit["typical"][j]) if fit["scale"]=="portion_scaled" else 1.
        radius = cal["q"]["adapted"][j]*factor
        np.testing.assert_allclose([f["point"], f["lower"], f["upper"]],
            [p, max(0,p-radius), p+radius], atol=.12, rtol=1e-4)
    print(f"PASS: 225 dish IDs and JPEG hashes, 20 quantiles and metric rows, {len(misses)} missed fields, live RGB intervals and calibration replacement.")


if __name__ == "__main__":
    main()
