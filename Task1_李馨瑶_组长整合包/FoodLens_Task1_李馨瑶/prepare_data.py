"""Safely unpack the supplied Food500Cap_ready archive and rebuild its train-only vocab."""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        files = [m for m in zf.infolist() if not m.is_dir()]
        roots = {PurePosixPath(m.filename).parts[0] for m in files}
        if len(roots) != 1:
            raise ValueError(f"Expected a single top-level dataset directory, got {roots}")
        prefix = next(iter(roots))
        for i, member in enumerate(files, 1):
            parts = PurePosixPath(member.filename).parts
            if not parts or parts[0] != prefix or ".." in parts or "\\" in member.filename:
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
            rel = Path(*parts[1:])
            if not rel.parts:
                continue
            target = (output / rel).resolve()
            if output not in target.parents:
                raise ValueError(f"ZIP path escapes output directory: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            if i % 1000 == 0 or i == len(files):
                print(f"Extracted {i}/{len(files)} entries", flush=True)

    ann = output / "annotations" / "train.jsonl"
    counts: collections.Counter[str] = collections.Counter()
    n = 0
    with ann.open(encoding="utf-8") as f:
        for line in f:
            counts.update(json.loads(line)["caption"].lower().split())
            n += 1
    words = sorted((w for w, c in counts.items() if c >= 5), key=lambda w: (-counts[w], w))
    vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    vocab.update({w: i + 4 for i, w in enumerate(words)})
    vocab_path = output / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image_and_metadata_entries": len(files), "train_caption_rows": n,
                      "vocab_size": len(vocab), "min_count": 5,
                      "vocab_note": "Rebuilt from train.jsonl; archive ZIP omitted vocab.json."},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
