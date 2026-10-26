import sys
import json
from predict import predict_nutrition


def main():
    if len(sys.argv) < 2:
        print("用法: python demo_task2.py <图片路径>")
        print("示例: python demo_task2.py images/test/dish_1556575327.png")
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict_nutrition(image_path)

    output = {"image": image_path, **result}
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()