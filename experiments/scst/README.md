# V4 SCST 正式预算实验

本目录承接scst_pilot的33步流程验证，独立实现固定预算训练、继续CE对照、验证选点、全量测试及视觉实体审计。结果完成后见[实验报告](REPORT.md)。所有模型权重与特征缓存在results/local_cache，不进入Git。

## 方法与数据

- 共同起点：正式full_v4/best_v4.pt及对应torchvision ViT-B/32 ImageNet权重。
- 固定种子20261127，训练2048图×3轮，batch size 8，每臂768步；Adam lr=1e-6，梯度裁剪1。
- 奖励：训练参考固定DF的CIDEr-D；减去同模型贪心奖励。非ROUGE-L试跑奖励。
- 256张验证图选最佳轮，不含100张实体审计图；固定4938张测试图最终计分。
- 保留原CE，另加同预算继续CE臂。只有一个种子、固定子集预算，不宣称全量训练或跨种子稳定。
- 视觉审计标注来自创新一的助手辅助、用户确认记录，不是独立真人盲标。

## 环境与运行

Python3.11，CUDA12.4版PyTorch2.5.1，RTX4060 8GB。安装GPU版框架后安装requirements中其余依赖；不需要联网下载新模型。

先取得原始Food500Cap_ready完整图片与annotations、正式V4检查点、torchvision冻结骨干权重，以及创新一review/human_annotations.jsonl。它们没有随本仓库提供；路径通过参数显式传入，不依赖作者桌面位置。

```powershell
python run.py --data-root "D:/data/Food500Cap_ready" --checkpoint "D:/weights/best_v4.pt" --backbone "D:/weights/vit_b_32-d86f8d99.pth" --annotations "../reranking/review/human_annotations.jsonl" --out results
python analyze.py --out results
python predict.py "D:/data/example.jpg" --checkpoint "results/local_cache/scst_best.pt" --backbone "D:/weights/vit_b_32-d86f8d99.pth" --device cpu
```

运行时从本目录执行；也可从仓库根目录给出脚本全路径。只运行实现检查可加`--mode tests`并使用独立输出目录。

完整运行应使用新输出目录。相同协议下再次执行支持从完整轮边界的last恢复；若进程在一轮中途终止，程序以最后完成轮恢复，历史文件先备份至缓存，再剔除未完成轮残留记录，避免把重复日志当新增训练证据。不同参数必须使用新目录。跨设备或任意中断不保证逐位一致。

## 产物

- protocol.json：训练前固定协议、数据量、种子、模型哈希、环境和局限。
- *_inputs.jsonl：训练、选点验证、实体审计、测试的完整输入ID与参考。
- implementation_checks.json、data_audit.json：奖励/概率实现核对和输入检查。
- *_history.jsonl、*_epochs.json、*_completion.json：逐步训练、逐轮选点、最佳权重哈希及完成状态。
- *_test_predictions.jsonl、*_test_metrics.json：同协议全量预测和成绩。
- comparison.csv、test_paired_bootstrap.json、entity_metrics.json、cases.jsonl：比较、不确定性与失败分析。
- training_and_results.png：奖励、梯度、选点曲线和测试结果。

指标采用pycocoevalcap实现、小写空格分词；不包含PTB/Java，不称完整官方COCO流水线。原测试集历史已被查看，本轮如实记为回溯评价。

方法依据：[Self-Critical Sequence Training，CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.html)。model.py与v1_data.py沿用项目正式V4，entities.py沿用创新一固定词表；run.py与analyze.py为本轮实现。
