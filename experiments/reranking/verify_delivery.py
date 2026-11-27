"""Verify the unmodified delivery against the bundled SHA256 manifest; Python standard library only."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
expected=json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
fail=[]
for name,digest in expected.items():
 p=root/name
 if not p.is_file():fail.append(name+': missing');continue
 with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 if actual!=digest:fail.append(name+': SHA256 mismatch')
if fail:raise SystemExit('\n'.join(fail))
print(f'PASS: SHA256 verified {len(expected)} delivery files')
