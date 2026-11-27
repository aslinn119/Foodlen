"""
demo_task2.py —— Task 2 命令行演示：同一张菜品图，三模型（A / B / B-gen）输出对比

用法:
  python demo_task2.py <图片路径> [model_id]
    不传 model_id：同图跑全部三个模型，输出对比 JSON
    model_id ∈ {a, b, bgen}：只跑指定模型
"""
import sys
import json
from predict import predict_nutrition_verbose, MODEL_IDS


def main():
    if len(sys.argv) < 2:
        print("用法: python demo_task2.py <图片路径> [model_id]")
        print("示例: python demo_task2.py images/test/dish_1556575327.png")
        print("      python demo_task2.py images/test/dish_1556575327.png bgen")
        print(f"可选模型: {sorted(MODEL_IDS)}")
        sys.exit(1)

    image_path = sys.argv[1]
    model_id = sys.argv[2] if len(sys.argv) > 2 else None

    if model_id:
        r = predict_nutrition_verbose(image_path, model_id)
        print(json.dumps({"image": image_path, **r}, indent=2, ensure_ascii=False))
    else:
        results = [predict_nutrition_verbose(image_path, mid) for mid in ["a", "b", "bgen"]]
        output = {
            "image": image_path,
            "models": {
                r["model"]: {
                    "model_name": r["model_name"],
                    "nutrition": r["nutrition"],
                    "text": r["text"],
                    "parse_code": r["parse_code"],
                    "warnings": r["warnings"],
                }
                for r in results
            },
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
