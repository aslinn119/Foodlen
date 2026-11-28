# 营养极限挑战正式模块

刘杰2026年11月26日正式交付，11月28日检查并集成。本模块与[Task2必做模块](../../task2/README.md)并列，保留各自的模型、数据域和评价身份。源码、冻结配置、指标与结果按原字节收录；[原提交说明](00_提交说明.md)中的“包内权重”指完整原始ZIP，GitHub副本不含大权重。

## 从哪里看

- [OmniFood8K主实验报告](challenge_external/实验报告.md)：三基线同划分、5668校准/2368评估、逐字段MAE/MSE/RMSE、覆盖率/宽度与欠覆盖案例。
- [主实验完整对照表](challenge_external/results/metrics.csv)。
- [NutritionVerse餐盘适配](challenge_external/nutritionverse_extension/餐盘优化实验报告.md)：68/30/60/67四用途划分。
- [MetaFood3D结果](challenge_external/metafood_extension/metrics.json)：200/50/150/95四用途划分。
- [创新二区间校准入口](../calibration/README.md)：指向本模块实现，不另复制一份校准代码。
- [本次检查结论](本次检查结论.md)、[复核记录](integration_checks.json)。
- [补充数据与来源说明](challenge_external/data_supplement_v1/补充数据验收与使用说明.md)：保留逐样本标注、划分、来源和许可说明；720张原始RGB外置，逐图路径/哈希见[external_images.json](external_images.json)。

## 模型边界

主实验mean/ResNet18/CLIP复用既有基线，未在Omni上重训。CLIP的历史B权重SHA256为`d9b2dff994213e3f3685ea16f07a7a014e3cbf68600a6e4a7e28de3f10bdcd79`；它不是Task2 normalized B。新旧模型的分数、校准参数和食材F1不可互换。NutritionVerse、MetaFood分别有自己的适配头与校准版本。

根目录predict.py仅作历史身份追溯，相关生成式依赖未全附，不作为本模块运行入口。本模块主要返回五项整道菜营养值与区间；items生成和原课程食材评价继续由Task2承担。

## 运行与外置文件

先从原始正式ZIP恢复[external_weights.json](external_weights.json)的三个权重和[external_images.json](external_images.json)的720张图片到本目录对应路径，并逐项核对哈希：checkpoints/best_model.pt、checkpoints_lora_mt/best_model.pt、clip_model/pytorch_model.bin。不要复制Task2 normalized B来代替。用于适配的轻量head.npz文件已经收录。

依赖参考[原环境清单](requirements_final.txt)。本次复核并非该清单所有精确版本的空白安装验收：主实验和MetaFood在已有CPU环境通过；NutritionVerse完整复核在torch2.5.1+cu124、transformers4.46.3的RTX4060环境通过，旧CPU环境的单图输出超出原容差，未放宽阈值。展示餐盘模型优先使用已验证GPU环境。

从本模块目录执行：

```powershell
python -m challenge_external.verify_delivery
python -m challenge_external.verify_nutritionverse
python -m challenge_external.verify_metafood
python -m challenge_external.demo_intervals "你的图片.jpg" --model clip
```

桌面展示分别使用[打开营养展示.cmd](打开营养展示.cmd)和[打开餐盘优化展示.cmd](打开餐盘优化展示.cmd)。它们是Tk桌面窗口，尚未与项目Web整合；不应称为已经接入Web。

本次复核覆盖保存结果重算与真实单图接口，没有从8036张Omni原图重跑全量推理。若需全量重推理，仍需另取`0-OminiFood8k.zip`，按原说明放入本模块根目录。补充域720图保留在原始正式ZIP，按清单恢复后才能运行完整图像验证。课程原3258图缺口不因此解决。

## 如何解读结果

90%是各字段目标覆盖率，不保证每张图或五项联合90%。Omni CLIP重量92.74%、脂肪73.10%；NutritionVerse热量覆盖62/67，平均区间仍宽1928.61 kcal。主Omni实验和补充域实验分开报告，不把域适配收益写成原Task2全面提升。场景/对象互斥不等于已证实批次独立或交换性。

原报告、预登记文件中的“尚未运行”等状态为历史记录，当前状态以正式结果、本页和复核记录为准。数据来源和许可说明按原交付保留，不为第三方数据另授新许可。
