"""Rebuild this extracted delivery using only local package files."""
import argparse,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=ROOT.parent/(ROOT.name+"_repacked.zip"));a=p.parse_args()
 files={f.relative_to(ROOT).as_posix():f for f in ROOT.rglob("*") if f.is_file() and "__pycache__" not in f.parts and f.name not in ["SHA256SUMS.json","package_verification.json"] and f.suffix!=".zip"}
 for name in ["best_v2.pt","last.pt","pretrained/resnet18-f37072fd.pth"]:
  if name not in files:raise FileNotFoundError(name)
 sums={name:hashlib.sha256(f.read_bytes()).hexdigest() for name,f in files.items()}
 manifest=ROOT/"SHA256SUMS.json";manifest.write_text(json.dumps(sums,ensure_ascii=False,indent=2),encoding="utf-8");files["SHA256SUMS.json"]=manifest
 if a.output.exists():raise FileExistsError("Use a new output path")
 with zipfile.ZipFile(a.output,"w",zipfile.ZIP_DEFLATED) as z:
  for name,f in sorted(files.items()):z.write(f,ROOT.name+"/"+name)
 with zipfile.ZipFile(a.output) as z:assert z.testzip() is None
 print(a.output)
if __name__=="__main__":main()
