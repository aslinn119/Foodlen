# 抽样复核 5 张图的选择记录

日期：2026-11-16（Asia/Shanghai）
范围：`deepseek_web_formal` 的 30 组图片，每组 A–D 四条候选；本次抽查 5 组 = 20 条。

## 一、3 张随机图

- 随机方法：Python `random.Random(20261116).sample(pool, 3)`，`pool = 01..30` 去掉两张问题图 `{03, 19}`，避免重复。
- 随机种子：`20261116`（可复现）
- 抽中：**08、14、23**

| 图号 | 参考描述（参考仅作辅助） | 说明 |
|---|---|---|
| 08 | black stuffed mushrooms with some meat stuffing inside and covered with a layer of cheese | 随机抽取 |
| 14 | brown stew chicken simmered with chicken cut into pieces, onions and chives in a pot | 随机抽取 |
| 23 | a plate of sliced pig's ear with three different dipping sauces on the side | 随机抽取 |

## 二、2 张有代表性的问题图

| 图号 | 参考描述 | 选取理由 |
|---|---|---|
| 03 | two pink galbis with two oval green leaves placed on a stainless steel net | **词表假阳性案例**：03A 候选写 "fish steak"，Task2 词表按 "steak" 与 Task2 输出的 "steak" 判为共有食材，但图中实为腌制牛肉排骨、且 "fish" 是候选自身的错误，词面重合不能证明肉类一致。 |
| 19 | arros negre added with some shrimps and mussels inside, garnished with red pepper slices | **Task2 明显过度预测案例**：Task2 对这张墨鱼汁黑米饭输出了 13 项蔬菜类（kale/spinach/cabbage/mushrooms…），既无米饭也无虾蟹；与候选、与画面都对不上，适合演示"两边各写了什么、谁漏了、哪些无法判断"。 |

## 三、说明

- 复核采用**独立子 Agent（DeepSeek V4.1 Flash，同一模型的不同上下文）**，未向其提供 DeepSeek 评分、模型身份或其他候选组内容；**不是真人复核**，只能作为代理复核/预填，报告若写"人工复核"需真人签认。
- 每张图由一个复核员看 A–D 四条，便于同图横向比较。
- 评分维度与 1–5 锚点沿用项目正式提示词的口径。
