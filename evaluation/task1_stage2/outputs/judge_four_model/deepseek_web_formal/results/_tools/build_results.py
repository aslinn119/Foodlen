# -*- coding: utf-8 -*-
"""Validate FoodLens formal judge raw replies and build normalized results.

Usage:
  python build_results.py            # validate all raw/*.txt, rebuild outputs
  python build_results.py --check 01 # validate one id only and print report

Reads : results/raw/<ID>.txt   (exact model reply, may be fenced)
Writes: results/<ID>.json      (verbatim model JSON object, re-serialized)
        results/manifest.json  (per-id metadata: status, attempts, errors)
        results/judge_scores.jsonl (one line per accepted candidate)
        results/_invalid.json  (ids that failed validation, with reasons)
"""
import json
import os
import re
import sys
import glob
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
BASE = os.path.dirname(RESULTS)
RAW = os.path.join(RESULTS, "raw")

SCORE_FIELDS = ["visual_consistency", "food_identification", "detail_accuracy", "language_quality"]
ARRAY_FIELDS = ["explicit_conflicts", "unverifiable_from_image",
                "visible_but_omitted", "suspected_ingredient_hallucinations"]
REQUIRED = ["candidate_id"] + SCORE_FIELDS + ["reasons"] + ARRAY_FIELDS

RUN_META = {
    "provider": "cc-switch-open-code-go",
    "model": "deepseek-v4.1-flash",
    "model_display": "DeepSeek V4.1 Flash 1017",
    "reasoning_effort": "max",
    "interface": "DSH subagent (OpenCode Go)",
    "prompt_version": "formal_web_v1",
}
TZ = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(TZ).isoformat(timespec="seconds")


def extract_json(text):
    """Extract the single JSON object from a model reply (handles fences/prose)."""
    t = text.strip()
    if t.startswith("\ufeff"):
        t = t[1:]
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip()
    try:
        return json.loads(t), None
    except Exception:
        pass
    # fall back: first balanced {...} block
    start = t.find("{")
    if start < 0:
        return None, "no '{' found in reply"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = t[start:i + 1]
                try:
                    return json.loads(blob), None
                except Exception as e:
                    return None, "balanced block is not valid JSON: %s" % e
    return None, "unbalanced braces"


def validate(obj, expected_id):
    problems = []
    if not isinstance(obj, dict):
        return ["top-level is not a JSON object"]
    # candidate fields with exact type checks
    if obj.get("candidate_id") != expected_id:
        problems.append("candidate_id is %r, expected %r" % (obj.get("candidate_id"), expected_id))
    for f in SCORE_FIELDS:
        if f not in obj:
            problems.append("missing field %s" % f)
            continue
        v = obj[f]
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int):
            problems.append("%s is %r (%s), expected int 1-5 or null" % (f, v, type(v).__name__))
        elif not (1 <= v <= 5):
            problems.append("%s is %r, out of 1-5" % (f, v))
    r = obj.get("reasons")
    if not isinstance(r, dict):
        problems.append("reasons is %r, expected object" % type(r).__name__)
    else:
        for f in SCORE_FIELDS:
            if f not in r:
                problems.append("reasons missing key %s" % f)
            elif not isinstance(r[f], str) or not r[f].strip():
                problems.append("reasons.%s is not a non-empty string" % f)
    for f in ARRAY_FIELDS:
        if f not in obj:
            problems.append("missing field %s" % f)
        elif not isinstance(obj[f], list):
            problems.append("%s is %s, expected array" % (f, type(obj[f]).__name__))
        else:
            bad = [i for i, x in enumerate(obj[f]) if not isinstance(x, str)]
            if bad:
                problems.append("%s has non-string items at %s" % (f, bad))
    return problems


def read_prompt_meta(image_id, cid):
    """Pull reference/candidate strings out of the candidate prompt file."""
    path = os.path.join(BASE, image_id, cid + ".txt")
    ref = cand = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("参考描述："):
                    ref = line[len("参考描述："):].strip()
                elif line.startswith("候选描述："):
                    cand = line[len("候选描述："):].strip()
    return ref, cand, path


def load_manifest():
    p = os.path.join(RESULTS, "manifest.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"run": RUN_META, "created": now_iso(), "candidates": {}}


def main():
    only = None
    if "--check" in sys.argv:
        only = sys.argv[sys.argv.index("--check") + 1]

    man = load_manifest()
    man["run"] = RUN_META
    man["updated"] = now_iso()
    man.setdefault("candidates", {})

    raw_files = sorted(glob.glob(os.path.join(RAW, "*.txt")))
    rows, invalid = [], []
    ok, failed = 0, 0
    for rp in raw_files:
        cid = os.path.splitext(os.path.basename(rp))[0]
        if only and cid != only:
            continue
        image_id = cid[:2]
        with open(rp, encoding="utf-8-sig", errors="replace") as f:
            text = f.read()
        obj, err = extract_json(text)
        problems = [err] if err else validate(obj, cid)
        entry = man["candidates"].get(cid, {})
        entry.update({
            "candidate_id": cid, "image_id": image_id,
            "raw_response_path": os.path.relpath(rp, RESULTS).replace("\\", "/"),
            "raw_bytes": len(text.encode("utf-8")),
            "validated_at": now_iso(),
        })
        if problems:
            failed += 1
            entry["status"] = "invalid"
            entry["validation_errors"] = problems
            man["candidates"][cid] = entry
            invalid.append({"candidate_id": cid, "errors": problems})
            print("INVALID %s: %s" % (cid, "; ".join(problems)))
            continue
        ok += 1
        entry["status"] = "ok"
        entry["validation_errors"] = []
        man["candidates"][cid] = entry
        out = os.path.join(RESULTS, cid + ".json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        ref, cand, ppath = read_prompt_meta(image_id, cid)
        row = {
            "candidate_id": cid,
            "image_id": image_id,
            "image_path": os.path.join(BASE, image_id, "image.jpg"),
            "prompt_path": ppath,
            "reference": ref,
            "candidate": cand,
            "visual_consistency": obj["visual_consistency"],
            "food_identification": obj["food_identification"],
            "detail_accuracy": obj["detail_accuracy"],
            "language_quality": obj["language_quality"],
            "reasons": obj["reasons"],
            "explicit_conflicts": obj["explicit_conflicts"],
            "unverifiable_from_image": obj["unverifiable_from_image"],
            "visible_but_omitted": obj["visible_but_omitted"],
            "suspected_ingredient_hallucinations": obj["suspected_ingredient_hallucinations"],
            "judge_model": RUN_META["model_display"],
            "judge_provider": RUN_META["provider"],
            "reasoning_effort": RUN_META["reasoning_effort"],
            "status": "ok",
            "human_review": None,
        }
        rows.append(row)

    if not only:
        rows.sort(key=lambda r: r["candidate_id"])
        jl = os.path.join(RESULTS, "judge_scores.jsonl")
        with open(jl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(RESULTS, "_invalid.json"), "w", encoding="utf-8") as f:
            json.dump(invalid, f, ensure_ascii=False, indent=2)
        man["summary"] = {
            "valid": ok, "invalid_in_raw": failed,
            "expected_total": 120, "built_at": now_iso(),
        }
    with open(os.path.join(RESULTS, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print("valid=%d invalid=%d (rows=%d)" % (ok, failed, len(rows)))


if __name__ == "__main__":
    main()
