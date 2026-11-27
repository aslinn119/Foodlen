# FoodLens 任务二 · 11 月 9 日最终交付修订版

安装 requirements_final.txt 中对应环境的依赖后，在本目录运行 python predict.py，即可用包内图片离线调用 A、B 和 B-gen。程序接口为 predict.predict_nutrition(image_path, model_id)。完整数据集评测需要原任务二图片放入 annotations 所指的路径。重新训练 B 需要另备原 CLIP 预训练底座。

模型说明见 MODELS.json。最终结果、训练选型及历史测试集的使用情况见 final_results_v3_20261119/最终修复报告.md。freeze.json 记录原评测环境；本次发布仅调整离线加载、路径和文档。历史阶段材料记录当时结果。

