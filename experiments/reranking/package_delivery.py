"""Package innovation-one source, models, fixed data and results, excluding runtime caches."""
from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parent
exclude={'__pycache__','local_cache','model_cache','reproduction','hf_cache'}
files=sorted(p for p in root.rglob('*') if p.is_file() and not exclude.intersection(p.relative_to(root).parts) and p.name!='SHA256SUMS.json')
assert not any(any(k in p.name.lower() for k in ['scst','rl_last','rl_history','ce_control','rl_after']) for p in files)
sums={}
for p in files:
 with p.open('rb') as f:sums[p.relative_to(root).as_posix()]=hashlib.file_digest(f,'sha256').hexdigest()
(root/'SHA256SUMS.json').write_text(json.dumps(sums,ensure_ascii=False,indent=2),encoding='utf-8')
files.append(root/'SHA256SUMS.json')
target=root.parent/'创新一_源码模型与实验结果_用户确认版_2026-11-21.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for p in files:z.write(p,p.relative_to(root.parent).as_posix())
with zipfile.ZipFile(target) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(files)
print(json.dumps({'archive':str(target),'file_count':len(files),'bytes':target.stat().st_size,'crc_verified':True},ensure_ascii=False),flush=True)
