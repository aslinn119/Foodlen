# 创新一交付包 2026年11月21日

本包仅包含“视觉证据约束的V4候选描述重排序”，已将之前相关成果一并整理；没有包含SCST训练入口、强化学习断点或奖励日志。

请先读REPORT.md。工程、首轮实测与本次用户确认记录已完成。用户表示已看完且与草稿一致，授权助手代录100行；已重算确认版指标。属于“助手辅助标注，经用户检查确认”，不是独立人工盲标。

## 文件

- pipeline.py / common.py：候选生成、CLIP评分和验证集选系数；model.py / v1_data.py为配套原始V4模型与数据变换。
- evaluate_text.py：COCO评价实现复算及5000次配对bootstrap。
- entities.py / evaluate_entities.py：食材同义词、可见证据统计、敏感性分析及人工签核导入。
- summarize.py / make_figures.py：汇总报告与图表。
- results：此前200×4候选、CLIP分数、最终选择、原始小样本指标及本轮补充的文本/食材指标、逐图证据、图表和配置。
- review：100图助手标注源文件、逐图标注、10张编号拼图、用户确认CSV和三方法对照CSV。
- assets/images：仅包含本轮200张验证图片，保留val/菜名/图片名路径，不是完整Food500Cap数据集。
- weights/v4_full.pt：原正式V4解码器与冻结ImageNet ViT合并权重，包含词表和配置；不是新训练模型。
- weights/clip：完整本地CLIP模型和分词/图像处理文件，支持断网运行。
- verification：3张图的独立完整权重推理、离线CLIP核对记录。
- SHA256SUMS.json：交付文件内容校验，verify_delivery.py可以检查。

## 环境

Python3.11，CPU试验版本见requirements.txt。解压到任意目录后，在Innovation1上一级打开PowerShell：

```powershell
python -m venv .venv-innovation1
& .venv-innovation1/Scripts/python.exe -m pip install -r Innovation1/requirements.txt
$env:HF_HOME = (Join-Path (Get-Location) 'Innovation1/local_cache')
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
```

Python虚拟环境和安装包缓存不在ZIP中，首次安装依赖需要联网。安装好上述库后，模型和图片均在包内，推理无需下载外部权重。CPU为本轮正式记录环境；CUDA可用时生成脚本会自动使用GPU，不能假定不同设备逐位相同。

## 重算已保存成绩

```powershell
& .venv-innovation1/Scripts/python.exe Innovation1/evaluate_text.py
& .venv-innovation1/Scripts/python.exe Innovation1/evaluate_entities.py
& .venv-innovation1/Scripts/python.exe Innovation1/summarize.py
& .venv-innovation1/Scripts/python.exe Innovation1/make_figures.py
```

上述命令复算已保存预测，不是重新完成200张图片推理。默认实体评分使用助手标注；脚本保留现有human_review.csv，不会重置已填写的人工字段。

## 从图片重新运行完整200图

```powershell
& .venv-innovation1/Scripts/python.exe Innovation1/pipeline.py generate --out Innovation1/reproduction
& .venv-innovation1/Scripts/python.exe Innovation1/pipeline.py clip --out Innovation1/reproduction
& .venv-innovation1/Scripts/python.exe Innovation1/pipeline.py rerank --out Innovation1/reproduction
& .venv-innovation1/Scripts/python.exe Innovation1/evaluate_text.py --results Innovation1/reproduction
```

第一次使用新的输出目录；generate遇到已有候选文件会拒绝覆盖，防止混入其他模型结果。CLI默认路径均相对源码目录，与原作者桌面路径无关。--checkpoint需要带full_model_state的完整模型权重；不能直接替换为原decoder-only checkpoint。--data-root应指向含val子目录的图片根目录。--clip-model可覆盖本地CLIP目录。

## 人工验收

按ENTITY_PROTOCOL.md真实复核review/human_review.csv，所有100行完成并签名日期后运行：

```powershell
& .venv-innovation1/Scripts/python.exe Innovation1/evaluate_entities.py --human-csv Innovation1/review/human_review.csv
```

输出独立entity_metrics_human.json和entity_per_image_human.jsonl。当前CSV按用户在聊天中的明确确认完成代录，姓名栏为“用户（本聊天确认，助手代录）”；未冒用真实姓名。独立结果entity_metrics_human.json已重算并更新报告，保留user_confirmation.json说明来源和授权。没有独立双人盲审。

## 检查

```powershell
& .venv-innovation1/Scripts/python.exe Innovation1/checks.py
python Innovation1/verify_delivery.py
```

SHA256用于交付原件；如果重跑后报告/指标有浮点差异或人工修改CSV，内容哈希不一致是预期的，应另存新的版本。本包包含模型与小样本原图，体积约1GB，请勿直接提交普通Git仓库。图片来源为本地Food500Cap_ready样本，仅为本次实验复现而附。

原先6图定性记录以review/initial_visual_review_6.json保留，已移除与本任务无关的输出列；本轮100图协议更严格区分“无法确认”，旧记录只作历史，不用于计算新实体指标。

若需从原交付材料重建完整V4，可运行export_model.py --checkpoint 原best_v4.pt --backbone 原vit_b_32-d86f8d99.pth --output 新full.pt；本包已经完成合并，不需要重新导出。

本次用户原话：“我看完了一模一样，你帮我填一下”。因此只代录确认一致的草稿，不声称新增独立标签。
