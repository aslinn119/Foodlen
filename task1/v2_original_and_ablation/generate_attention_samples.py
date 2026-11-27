"""Render word-aligned attention overlays for selected held-out examples."""
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "v2_attention"))
from v2_attention.predict_caption import CaptionPredictor


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--indices",type=int,nargs="*",default=[190,1376,4683])
    a=p.parse_args()
    rows=[json.loads(line) for line in (a.root/"annotations"/"test.jsonl").open(encoding="utf-8")]
    predictor=CaptionPredictor(a.checkpoint)
    out=BASE/"report_figures"/"task1"/"attention_samples"
    for index in a.indices:
        row=rows[index];folder=out/f"sample_{index:04d}"
        image=a.root/row["file_name"]
        caption=predictor.visualize(image,folder)
        meta=json.loads((folder/"caption.json").read_text(encoding="utf-8"))
        meta.update(index=index,category=row["cat"],reference=row["caption"])
        (folder/"caption.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({"index":index,"category":row["cat"],"reference":row["caption"],"hypothesis":caption},ensure_ascii=False))


if __name__=="__main__":main()
