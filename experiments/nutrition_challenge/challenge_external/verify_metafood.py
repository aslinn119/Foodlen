"""Verify MetaFood images, held-out predictions, intervals, and live inference."""
import json
import math

import numpy as np

from challenge_external.metafood_experiment import DATA, KEYS, OUT, predict_metafood_intervals
from challenge_external.runtime import sha


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    protocol = read_json(OUT / "protocol.json")
    artifact = read_json(OUT / "calibration.json")
    freeze = read_json(OUT / "freeze.json")
    assert artifact["selected"] == freeze["selected"] == min(freeze["validation_scores"], key=freeze["validation_scores"].get)
    assert artifact["dependencies"] == freeze["dependencies"]
    assert all(artifact["dependencies"][p] == digest for p, digest in protocol["dependencies"].items())
    version = "metafood-abs90-" + sha(OUT / "calibration_predictions.json")[:12]
    assert artifact["version"] == version and artifact["target_coverage"] == .9
    splits = {}
    for role, count in (("train", 200), ("validation", 50), ("calibration_reserved", 150), ("test_reserved", 95)):
        path = DATA / (role + ".jsonl")
        assert sha(path) == protocol["roles"][role], role
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == count
        for row in rows:
            assert row["role"] == role and sha(DATA / row["image"]) == row["image_sha256"], row["sample_id"]
        splits[role] = rows
    for key in ("object_id", "sample_id", "image_sha256"):
        assert len({row[key] for rows in splits.values() for row in rows}) == 495, key

    cal = read_json(OUT / "calibration_predictions.json")
    test = read_json(OUT / "evaluation_predictions.json")
    for role, records in (("calibration_reserved", cal), ("test_reserved", test)):
        assert records["ids"] == [row["sample_id"] for row in splits[role]]
        truth = [[row["target"][field] for field in KEYS] for row in splits[role]]
        np.testing.assert_array_equal(records["targets"], truth)
        assert set(records["points"]) == {"clip", "ridge"}
    rank = math.ceil(.9 * (len(cal["ids"]) + 1))
    assert rank == artifact["rank"] == 136 and artifact["n"] == 150
    reference_rows = read_json(OUT / "metrics.json")
    table = {(r["model"], r["field"]): r for r in reference_rows}
    assert len(reference_rows) == len(table) == 10
    failures = {}
    truth = np.array(test["targets"])
    for model in ("clip", "ridge"):
        scores = np.abs(np.array(cal["points"][model]) - cal["targets"])
        radius = np.sort(scores, axis=0)[rank - 1]
        np.testing.assert_allclose(radius, artifact["q"][model])
        points = np.array(test["points"][model])
        lower, upper = np.maximum(0, points - radius), points + radius
        covered = (lower <= truth) & (truth <= upper)
        for j, field in enumerate(KEYS):
            error = points[:, j] - truth[:, j]
            actual = [abs(error).mean(), np.square(error).mean(), covered[:, j].sum(),
                      len(error), covered[:, j].mean(), (upper - lower)[:, j].mean()]
            record = table[model, field]
            np.testing.assert_allclose(actual, [record[k] for k in ("mae", "mse", "hits", "n", "coverage", "mean_width")])
            for i in np.flatnonzero(~covered[:, j]):
                failures[model, test["ids"][i], field] = [points[i, j], lower[i, j], upper[i, j], truth[i, j]]
    recorded = read_json(OUT / "undercoverage.json")
    assert len(failures) == len(recorded)
    for r in recorded:
        expected = failures.pop((r["model"], r["id"], r["field"]))
        np.testing.assert_allclose(expected, [r[k] for k in ("point", "lower", "upper", "target")])
    assert not failures

    example = read_json(OUT / "interface_example.json")
    live = predict_metafood_intervals(DATA / splits["test_reserved"][0]["image"])
    assert live["calibration_version"] == version and live["target_coverage"] == .9
    for key in ("model_version", "calibration_version", "target_coverage", "coverage_scope"):
        assert live[key] == example[key]
    for j, field in enumerate(KEYS):
        value = live["fields"][field]
        p = test["points"][artifact["selected"]][0][j]
        q = artifact["q"][artifact["selected"]][j]
        expected = [p, max(0, p - q), p + q]
        assert value["unit"] == example["fields"][field]["unit"] == ("kcal" if j == 1 else "g")
        for output in (value, example["fields"][field]):
            np.testing.assert_allclose([output[k] for k in ("point", "lower", "upper")], expected, atol=.12, rtol=1e-4)
    print(f"PASS: 495 unique objects and RGB hashes, 10 quantiles and metric rows, {len(recorded)} missed fields, live RGB intervals.")


if __name__ == "__main__":
    main()
