"""Rescore packaged predictions; requires no image dataset or training weights."""
import json
from pathlib import Path
from train import metrics
ROOT=Path(__file__).resolve().parent

def main():
 results={}
 recorded=json.loads((ROOT/"final_results.json").read_text(encoding="utf-8"))
 for split in ["val","test"]:
  rows=[json.loads(l) for l in (ROOT/f"{split}_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
  refs=[json.loads(l) for l in (ROOT/f"data/annotations/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
  ids=[r["file_name"] for r in rows];rids=[r["file_name"] for r in refs]
  if len(set(ids))!=len(ids) or len(set(rids))!=len(rids) or set(ids)!=set(rids):raise ValueError("ID mismatch")
  by_id={r["file_name"]:r for r in refs}
  if any(r["reference"]!=by_id[r["file_name"]]["caption"] for r in rows):raise ValueError("reference mismatch")
  results[split]={"normalized":metrics(rows),"legacy_whitespace":metrics(rows,False)}
  for protocol in results[split]:
   got=results[split][protocol];expected=recorded[split][protocol]
   assert max(abs(a-b) for a,b in zip(got["bleu"],expected["bleu"]))<1e-8
   assert abs(got["rouge_l"]-expected["rouge_l"])<1e-8 and abs(got["cider_d"]-expected["cider_d"])<1e-8
 (ROOT/"rescored_results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
 print("PASS: both complete splits match recorded metrics")
if __name__=="__main__":main()
