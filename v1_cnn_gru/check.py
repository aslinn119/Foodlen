# -*- coding: utf-8 -*-
# 数据集目录结构检查脚本：确认数据集路径存在，并预览目录结构和标注文件内容
from pathlib import Path  # 面向对象的路径处理库

# 你的数据集目录
DATASET_ROOT = Path(
    r"C:\Users\刘九尔\Desktop\神经网络与深度学习\Food500Cap_ready"
)

print("=" * 60)
print("数据集目录是否存在：", DATASET_ROOT.exists())  # 先确认路径对不对
print("=" * 60)

if not DATASET_ROOT.exists():
    print("路径不存在，请检查文件夹名称和路径。")
else:
    print("\n第一层目录和文件：")

    # 遍历根目录下的第一层内容，区分文件夹和文件打印出来
    for item in sorted(DATASET_ROOT.iterdir()):
        if item.is_dir():
            print(f"[文件夹] {item.name}")
        else:
            print(f"[文件]   {item.name}")

    print("\n可能的标注文件：")

    # 常见的标注文件扩展名
    extensions = {".json", ".jsonl", ".csv", ".txt", ".tsv"}
    annotation_files = []

    # 递归遍历整个数据集目录，找出所有上述扩展名的文件
    for path in DATASET_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            annotation_files.append(path)

    if annotation_files:
        for path in annotation_files:
            # 打印相对路径和文件大小
            print(f"\n文件：{path.relative_to(DATASET_ROOT)}")
            print(f"大小：{path.stat().st_size / 1024:.2f} KB")

            # 只读取文本类标注文件的开头，避免一次性加载大文件
            if path.suffix.lower() in {".json", ".jsonl", ".csv", ".txt", ".tsv"}:
                try:
                    # utf-8-sig 可自动去掉文件开头的 BOM
                    with path.open("r", encoding="utf-8-sig") as f:
                        for i in range(3):  # 只预览前 3 行
                            line = f.readline()
                            if not line:  # 文件不足 3 行就提前结束
                                break
                            print(f"  第{i + 1}行：{line[:500].rstrip()}")  # 每行最多显示 500 字符
                except Exception as e:
                    print("读取失败：", e)
    else:
        print("没有找到常见格式的标注文件。")
