# V4 首轮实验

本目录为独立实验副本。结果和结论见 REPORT.md，未修改桌面原始 V4 模型。需要正式 V4 checkpoint、原 ImageNet ViT-B/32 缓存和 Food500Cap_ready 图片集。默认路径在 experiment.py 中，可用 --checkpoint / --data-root 指定。

在 Foodlen 工作目录使用 Python 3.11 + torch + torchvision + Pillow（本轮默认 Python 的 torch 2.2.2 CPU）。CLIP 使用现有 .venv-task2-check 环境的 transformers。没有安装或修改全局包。

复现命令（PowerShell）：

```powershell
python V4_experiments/experiment.py prepare --out V4_experiments/reproduction
python V4_experiments/experiment.py candidates --out V4_experiments/reproduction
$env:HF_HOME = (Join-Path (Get-Location) 'V4_experiments/model_cache')
$env:HF_HUB_OFFLINE = '1'
& .venv-task2-check/Scripts/python.exe V4_experiments/experiment.py clip --out V4_experiments/reproduction
python V4_experiments/experiment.py rl --steps 32 --out V4_experiments/reproduction
python V4_experiments/experiment.py rl --steps 33 --resume V4_experiments/reproduction/rl_last.pt --out V4_experiments/reproduction
python V4_experiments/check_decoding.py
```

重新试验使用新 --out 目录；candidates 会从已有图片 ID 继续写入，不能在同一目录更换模型/配置。prepare 在新目录生成固定划分。summarize.py 汇总本轮 results；不传入其他目录。

本轮产物：candidates.jsonl（200x4）、clip_candidates.jsonl、reranked.jsonl、decode_metrics.json、rerank_metrics.json、ce_control.jsonl、rl_after.jsonl、rl_history.jsonl、rl_last.pt（模型/优化器/随机状态）、manifest.json（SHA256/ID/配置）、visual_review_6.json、ingredient_review.csv。

ROUGE-L 是本轮 SCST 的最小奖励，不是 CIDEr 优化。实验只跑 33 个更新步，未完成64图的完整一轮。食材核查完成6图，其他94张审计图的标注仍待完成；不要把空白表当作通过。原 CE checkpoint 作为共同起点保留在桌面路径，未另复制大权重。

model.py/v1_data.py 源于桌面正式 V4；caption_metrics.py/bleu.py 复用本项目已有本地评价口径。CLIP 缓存约600MB，rl_last.pt 含优化器较大，不应提交普通 Git 仓库。
