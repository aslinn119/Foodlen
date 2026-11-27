import hashlib,time,urllib.request,urllib.error
from pathlib import Path
url="https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp311-cp311-win_amd64.whl"
expected="6c8a7003ef1327479ede284b6e5ab3527d3900c2b2d401af15bcc50f2245a59f"
root=Path("D:/FoodLens_V2_Runtime/wheels");root.mkdir(exist_ok=True)
p=root/"torch.part";total=2510838912
# Discover and verify total from authoritative Content-Range rather than assume it.
failures=0
while True:
 start=p.stat().st_size if p.exists() else 0
 req=urllib.request.Request(url,headers={"Range":f"bytes={start}-{start+16*1024*1024-1}"})
 try:
  with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=45) as r:
   cr=r.headers.get("Content-Range","")
   if r.status!=206 or not cr.startswith("bytes "+str(start)+"-"):raise RuntimeError("server did not honor byte range: "+str(r.status)+" "+cr)
   total=int(cr.split("/")[-1]);written=0
   with p.open("ab") as out:
    while True:
     block=r.read(1024*1024)
     if not block:break
     out.write(block);written+=len(block)
  if not written:raise RuntimeError("empty range")
  failures=0;print(p.stat().st_size,total,flush=True)
  if p.stat().st_size==total:break
  if p.stat().st_size>total:raise RuntimeError("oversize response")
 except Exception as e:
  failures+=1;print("retry",start,type(e).__name__,str(e)[:150],flush=True)
  if failures>=40:raise
  time.sleep(2)
actual=hashlib.sha256(p.read_bytes()).hexdigest()
if actual!=expected:raise RuntimeError("HASH MISMATCH: "+actual)
dest=root/"torch-2.5.1+cu124-cp311-cp311-win_amd64.whl";p.replace(dest)
print("VERIFIED",dest,flush=True)
