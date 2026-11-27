# FoodLens 同图食材一致性初筛

刘杰的提交包包含模型 A、B 和 B-gen 的已有预测，标识为 `dish_id`。未找到与 Task1 盲评图片的匹配记录。本次使用提交包中的模型 B（CLIP + LoRA，默认推理设置）对盲评清单的 30 张原图重新推理，生成 30 条营养 JSON。依完整 `file_name` 与四模型的 120 条 Caption 精确对齐，每张图对应四条 Caption。

只提取 Task2 模型 B 食材词表能识别的英文词或短语，忽略大小写和标点，按最长短语匹配，并兼容部分单复数。对每条 Caption 保存两边共有、仅 Caption 有、仅 Task2 有的食材。一张图的 Task2 结果没有返回食材，仍保留在分母中。

| Caption 模型 | 图片数 | Caption 命中至少一项词表 | 两边至少共享一项 |
|---|---:|---:|---:|
| V1 重训消融 | 30 | 21 | 3 |
| V2 注意力 | 30 | 19 | 3 |
| V3.2 layer4 | 30 | 21 | 3 |
| V4 full | 30 | 24 | 5 |

例 03A 的 Caption 写了 “fish steak”。程序识别到 “steak” 与 Task2 输出重合，但这个词在句子里描述鱼排，单凭重合词无法确认肉类一致。未被词表覆盖的词和同义表达也可能造成漏匹配。因此这些数值仅作为人工复核线索，不能用来估算幻觉率，也不能当作营养精度或带标注的联合测试成绩。

营养预测见 `task2_model_b_same_images.jsonl`，逐条比对见 `ingredient_comparison.jsonl`，统计见 `ingredient_summary.csv`。推理使用刘杰包内的完整权重严格加载，视觉骨干结构来自 [CLIP ViT-B/32 配置](https://huggingface.co/openai/clip-vit-base-patch32/blob/main/config.json)。同图分析按固定图片清单运行，未使用真实营养标签。
