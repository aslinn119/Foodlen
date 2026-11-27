# 任务二验收与复现

在完整提交包目录按 requirements_final.txt 配置环境，运行 python demo_task2.py demo_images/dish_1556575327.png 可查看同一张图的三个模型输出。predict.predict_nutrition(image_path, model_id) 支持 a、b、bgen。

模型 A 使用受 JSON 语法和食材词表约束的解码器。模型 B 使用 checkpoints_lora_mt_normalized/best_model.pt 内保存的阈值，超出 16 项时按置信度截取。B-gen 使用历史 B 编码器。五个字段的名称、单位和约束见 schema.json。

验证和测试的逐样本预测、分字段指标、训练选型记录及报告均在 final_results_v3_20261119/。训练、验证和测试的 dish_id 无交集。测试数据曾用于历史对照，本版成绩属于回溯评估。运行 python -m unittest test_task2_corrections -v 可重跑回归测试。

11 月 9 日的挑战基线入口为 predict.predict_challenge_baseline(image_path, model_id)，支持 mean、resnet18、clip。这里的 clip 使用历史 B 权重，以复现基线。其接口返回五个点预测。

历史版实验说明和 Word 稿在 historical_reports/ 下，其中出现的阈值、指标和相对路径对应当时版本。
