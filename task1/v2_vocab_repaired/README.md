# V2 词表修复版完整交付

本包对应2026年11月18日启动、11月20日完成的V2重训。训练25轮，最佳权重为第24轮。训练1—12轮使用CPU，13—25轮从完整断点续训至RTX4060 GPU；不是一次全程GPU单设备训练。

## 能否替代旧V2

可作为系统中新版V2接入，但必须整体使用本包模型代码、词表、权重和图像变换。不要只覆盖旧best_v2.pt或旧vocab.json。新词表4837项，旧词表2681项，参数形状和token ID不同。

效果不支持全面替代：同一原空格分词的测试BLEU-4从5.2914%降至5.1885%，ROUGE-L从0.2695升至0.2716，CIDEr-D从0.3713升至0.4044；含UNK描述559条降至0，重复三词片段1005条降至764条。建议命名为v2_vocab_repaired，保留旧V2供对照；论文/报告不能写成全面优于旧版。没有UNK也不等于食材识别正确。

## 包含的材料

- train.py、model_v2.py、predict.py：可迁移的训练/模型/单图入口；新增evaluate_saved.py可不依赖完整图片复算保存成绩。
- training_source_original：当时实际运行的原始训练、模型、推理、打包源码，保留原路径以审计；根目录训练仅调整数据/预训练权重/基线文件路径，未改变学习算法。
- best_v2.pt：第24轮最佳模型，完整视觉骨干与解码器，可独立推理；checkpoint内含配套词表。
- last.pt：第25轮完整续训断点，含优化器、学习率调度器、随机状态和训练历史；不要将其误认为最佳推理权重。
- pretrained/resnet18-f37072fd.pth：从头重训所用ImageNet视觉预训练权重；单图推理不需要它。
- vocab.json、config.json、CPU/GPU日志、训练曲线、完整验证/测试预测、指标、新旧对照、20张验证案例和校验记录。
- data/annotations：train/val/test正式标注；baseline：旧V2预测，供同口径比较；examples：一张单图示例。

不包含完整Food500Cap图片集、Python解释器、CUDA安装包及环境缓存。download_cuda.py是历史环境下载辅助脚本，路径仅适用于原机器，不是训练或推理的必要依赖。

## 环境

Python3.11。GPU实际环境torch2.5.1+cu124与torchvision0.20.1+cu124；CPU初始阶段torch2.2.2。requirements.txt列出核心依赖，环境锁定记录见environment_versions.json；未宣称已在空白机器重装验证。

CPU安装示例：python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

GPU安装示例：python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

安装其余依赖：python -m pip install numpy==1.26.4 Pillow pycocoevalcap==1.2 matplotlib

## 单图推理与分数复算

解压后在包目录运行：

    python predict.py examples/sample.jpg --device cpu
    python evaluate_saved.py

接口：from predict import predict_caption；predict_caption(image_path, device="cpu")返回caption、model_id与epoch。默认优先GPU，支持CPU。单图无需完整数据和外部旧V2目录。

## 从头重训与续训

图片应按data/annotations中file_name放置。可以将完整图片目录放入data下，也可以在PowerShell设置：

    $env:FOODLENS_DATA_ROOT = '完整Food500Cap_ready路径'

该目录应同时包含annotations和train/val/test图片。仅标注不足以重训。预训练权重默认使用包内pretrained目录；可通过FOODLENS_RESNET18指定其他等价文件。

为避免覆盖这次交付结果，先复制整个交付目录到一个新的实验目录再运行：

    python train.py

从last.pt继续到总计30轮：

    python train.py --resume --epochs 30

续训会读取last中词表、模型、优化器和调度器，并重建新实验记录；本次历史val/test划分应保持不变。Windows多进程增强随机状态未逐worker保存，跨设备/中断续训不保证逐位复现。评测只在训练结束后执行；不要根据test继续调参。

根目录脚本与training_source_original存在的区别仅为迁移和接口支持，原有训练权重不变。SHA256SUMS.json记录压缩包内文件哈希。

## 可迁移性与历史路径

根目录build_full_v2_delivery.py可仅用解压目录内文件重新打包。training_source_original内脚本及download_cuda.py属于历史审计材料，不作为迁移后的运行入口；其中桌面、缓存和D盘路径是原运行环境记录。config、日志与runtime_history内路径也保留原记录。推理、保存结果复算和重打包不读取原工作区。重新训练仍须通过FOODLENS_DATA_ROOT提供完整图片；这是显式的数据依赖。独立解压检查见standalone_verification.json。package_verification.json是压缩包外的签收记录，不放入ZIP以避免自身哈希循环。
