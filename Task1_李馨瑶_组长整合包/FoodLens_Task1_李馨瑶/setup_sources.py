"""Preserve the supplied V1 source, and copy available baseline logs."""
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r'C:\Users\lenovo\Documents\xwechat_files\wxid_a37fxjd7zrzw22_44b7\msg\file\2026-10')
with zipfile.ZipFile(SOURCE / 'v1_cnn_gru(1).zip') as archive:
    for info in archive.infolist():
        if info.is_dir() or '__pycache__' in info.filename:
            continue
        target = ROOT / info.filename
        if not target.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(info.filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(info))
baseline = ROOT / 'baseline_reference'
baseline.mkdir(exist_ok=True)
for name in ['history.json', 'samples_test_greedy.txt']:
    source = SOURCE / name
    if source.exists():
        shutil.copy2(source, baseline / name)
print('V1 source and available reference logs preserved')
