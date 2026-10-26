
from pathlib import Path
import json

# 数据集根目录
DATASET_ROOT = Path(
    r"C:\Users\刘九尔\Desktop\神经网络与深度学习\Food500Cap_ready"
)

# 检查三个数据划分
for split in ["train", "val", "test"]:
    annotation_file = DATASET_ROOT / "annotations" / f"{split}.jsonl"

    total = 0
    missing = []

    with annotation_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            image_path = DATASET_ROOT / record["file_name"]
            total += 1

            if not image_path.is_file():
                missing.append(record["file_name"])

    print(f"\n{split} 集")
    print(f"标注条数：{total}")
    print(f"缺失图片数：{len(missing)}")

    if missing:
        print("缺失图片示例：")
        for name in missing[:5]:
            print(" ", name)
