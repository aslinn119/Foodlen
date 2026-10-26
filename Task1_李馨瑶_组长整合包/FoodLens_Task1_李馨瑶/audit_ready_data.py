"""Audit split annotations, image paths, duplicates, captions, and vocabulary."""
import collections
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "Food500Cap_ready"
OUT = Path(__file__).resolve().parent / "checks" / "ready_data_audit.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    result = {"root": str(ROOT), "splits": {}, "missing_images": [], "duplicate_paths": [],
              "caption_empty": [], "category_counts": {}, "vocab_size": 0}
    seen = set()
    cats = collections.Counter()
    for split in ("train", "val", "test"):
        path = ROOT / "annotations" / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        missing = [r["file_name"] for r in rows if not (ROOT / r["file_name"]).is_file()]
        duplicates = [r["file_name"] for r in rows if r["file_name"] in seen]
        seen.update(r["file_name"] for r in rows)
        result["missing_images"].extend(missing)
        result["duplicate_paths"].extend(duplicates)
        result["caption_empty"].extend(f"{split}:{i}" for i, r in enumerate(rows) if not r.get("caption", "").strip())
        cats.update(r["cat"] for r in rows)
        result["splits"][split] = {"rows": len(rows), "annotation_sha256": sha256(path)}
    vocab_path = ROOT / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    result["vocab_size"] = len(vocab)
    result["vocab_sha256"] = sha256(vocab_path)
    result["category_counts"] = {"count": len(cats), "most_common": cats.most_common(5)}
    result["image_count"] = len(seen)
    result["pass"] = (not result["missing_images"] and not result["duplicate_paths"]
                      and not result["caption_empty"] and result["vocab_size"] == 2681)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("category_counts",)}, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit("Food500Cap_ready data audit failed")


if __name__ == "__main__":
    main()
