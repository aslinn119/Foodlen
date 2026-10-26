
from pathlib import Path

# 你的数据集目录
DATASET_ROOT = Path(
    r"C:\Users\刘九尔\Desktop\神经网络与深度学习\Food500Cap_ready"
)

print("=" * 60)
print("数据集目录是否存在：", DATASET_ROOT.exists())
print("=" * 60)

if not DATASET_ROOT.exists():
    print("路径不存在，请检查文件夹名称和路径。")
else:
    print("\n第一层目录和文件：")

    for item in sorted(DATASET_ROOT.iterdir()):
        if item.is_dir():
            print(f"[文件夹] {item.name}")
        else:
            print(f"[文件]   {item.name}")

    print("\n可能的标注文件：")

    extensions = {".json", ".jsonl", ".csv", ".txt", ".tsv"}
    annotation_files = []

    for path in DATASET_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            annotation_files.append(path)

    if annotation_files:
        for path in annotation_files:
            print(f"\n文件：{path.relative_to(DATASET_ROOT)}")
            print(f"大小：{path.stat().st_size / 1024:.2f} KB")

            # 只读取文本类标注文件的开头，避免一次性加载大文件
            if path.suffix.lower() in {".json", ".jsonl", ".csv", ".txt", ".tsv"}:
                try:
                    with path.open("r", encoding="utf-8-sig") as f:
                        for i in range(3):
                            line = f.readline()
                            if not line:
                                break
                            print(f"  第{i + 1}行：{line[:500].rstrip()}")
                except Exception as e:
                    print("读取失败：", e)
    else:
        print("没有找到常见格式的标注文件。")
