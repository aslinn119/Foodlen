# 系统展示与联调

状态更新（2026年11月27日）：用户已告知李馨瑶完成Web；本轮按用户要求暂不接入，尚未检查其代码与真实模型链路。本目录保留此前字段草案与接口待办。

- [Web展示字段草案](../evaluation/task1_stage2/api_display_spec.md)
- [新版V2单图入口说明](../task1/v2_vocab_repaired/README.md)
- [营养模块说明](../task2/nutrition/README.md)

后续统一predict_caption(image, model_id)、predict_nutrition(image)的调用、设备与缺文件错误；接入上传、模型身份、caption、食材及单位、Judge状态、注意力与预测区间。至少保存5张共同输入的真实联调记录，再完成网页演示。当前目录不表示服务已实现。

11月28日新增两套独立的[营养挑战Tk桌面展示](../experiments/nutrition_challenge/README.md)。它们具有真实图片推理与区间输出，尚未接入李馨瑶的Web。
