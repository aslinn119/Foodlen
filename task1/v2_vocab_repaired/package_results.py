import json,csv,hashlib,random,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 results=json.loads((ROOT/"final_results.json").read_text(encoding="utf-8"));history=json.loads((ROOT/"history.json").read_text(encoding="utf-8"))
 table=[]
 for split in ["val","test"]:
  for model,normal,legacy in [("old_v2","old_v2_same_normalization","old_v2_legacy"),("new_v2","normalized","legacy_whitespace")]:
   for protocol,key in [("normalized_words",normal),("legacy_whitespace",legacy)]:
    m=results[split][key];table.append(dict(split=split,model=model,protocol=protocol,n=m["n"],BLEU1=m["bleu"][0],BLEU4=m["bleu"][3],ROUGE_L=m["rouge_l"],CIDEr_D=m["cider_d"],UNK_count=m["unknown_count"],repeated_trigram_count=m["repeated_trigram_count"]))
 with (ROOT/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True)
 axes[0].plot([h["epoch"] for h in history],[h["train_loss"] for h in history]);axes[0].set_ylabel("Training cross entropy");axes[0].grid(alpha=.2)
 axes[1].plot([h["epoch"] for h in history],[h["val"]["bleu"][3] for h in history],label="New V2 / full validation")
 axes[1].axhline(results["val"]["old_v2_same_normalization"]["bleu"][3],linestyle="--",color="gray",label="Old V2 / same normalization")
 axes[1].axvline(12.5,color="orange",linestyle=":",label="CPU -> GPU resume")
 axes[1].set_ylabel("Validation BLEU-4 (%)");axes[1].set_xlabel("Epoch");axes[1].legend();axes[1].grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/"training_curve.png",dpi=160);plt.close(fig)
 new=[json.loads(l) for l in (ROOT/"val_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
 old={r["file_name"]:r for r in [json.loads(l) for l in (ROOT/"baseline/V2_val_fresh.jsonl").read_text(encoding="utf-8").splitlines()]}
 sample=random.Random(42).sample(sorted(new,key=lambda r:r["file_name"]),20)
 (ROOT/"validation_cases20.json").write_text(json.dumps([dict(r,old_caption=old[r["file_name"]]["caption"]) for r in sample],ensure_ascii=False,indent=2),encoding="utf-8")
 text=(ROOT/"训练结果.md").read_text(encoding="utf-8")
 text+="\n\n训练设备：第1—12轮CPU（torch2.2.2），第13轮起RTX4060 Laptop GPU（torch2.5.1+cu124）；从第12轮完整训练断点恢复，保留优化器与调度器。跨设备续训不声称逐位可复现。\n"
 text+="\n词表覆盖改善和输出没有UNK不等于未见食物识别正确，仍需人工检查样例。重复三词统计仅为诊断，正常短语也可能重复。固定随机种子抽取20张验证样例见validation_cases20.json。\n"
 text+="\n训练曲线：\n\n![Training curves](training_curve.png)\n"
 (ROOT/"训练结果.md").write_text(text,encoding="utf-8")
 readme=(ROOT/"README.md").read_text(encoding="utf-8").replace("CPU训练，GPU依赖安装因自动审批超时未获执行。","已完成CPU到GPU续训，设备阶段详见runtime_history.json；GPU环境位于D:/FoodLens_V2_Runtime/venv。")
 (ROOT/"README.md").write_text(readme,encoding="utf-8")
 manifest={"results":results,"sha256":{}}
 for p in ROOT.iterdir():
  if p.is_file() and p.name not in ["last.pt","manifest.json"]:manifest["sha256"][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 (ROOT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
 dest=ROOT.parent/(ROOT.name+"_训练结果.zip")
 with zipfile.ZipFile(dest,"w",zipfile.ZIP_DEFLATED) as z:
  for p in ROOT.iterdir():
   if p.is_file() and p.name not in ["last.pt","download_cuda.py"]:z.write(p,ROOT.name+"/"+p.name)
 with zipfile.ZipFile(dest) as z:assert z.testzip() is None
 print(dest, dest.stat().st_size,flush=True)
if __name__=="__main__":main()
