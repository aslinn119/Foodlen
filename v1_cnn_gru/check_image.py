# -*- coding: utf-8 -*-
# 图片完整性检查：逐条核对标注里的图片文件是否真实存在于磁盘上
from pathlib import Path
import json

# 数据集根目录
DATASET_ROOT = Path(
    r"C:\Users\刘九尔\Desktop\神经网络与深度学习\Food500Cap_ready"
)

# 检查三个数据划分
for split in ["train", "val", "test"]:
    # 该划分对应的标注文件，如 annotations/train.jsonl
    annotation_file = DATASET_ROOT / "annotations" / f"{split}.jsonl"

    total = 0      # 标注总条数
    missing = []   # 缺失图片的文件名列表

    with annotation_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue  # 跳过空行

            record = json.loads(line)  # 每行是一条 JSON 记录
            # 拼接出图片的完整路径
            image_path = DATASET_ROOT / record["file_name"]
            total += 1

            if not image_path.is_file():
                missing.append(record["file_name"])  # 文件不存在则记入缺失列表

    # 打印该划分的统计结果
    print(f"\n{split} 集")
    print(f"标注条数：{total}")
    print(f"缺失图片数：{len(missing)}")

    if missing:
        print("缺失图片示例：")
        for name in missing[:5]:  # 最多展示前 5 个缺失文件名
            print(" ", name)
