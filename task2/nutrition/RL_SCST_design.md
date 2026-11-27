# B-gen RL SCST 可行性设计

## 目标

在现有 B-gen 自回归 JSON 解码器基础上，验证序列级奖励是否能改善生成文本质量，同时不牺牲 JSON 合法性和营养字段精度。该实验属于 P2，优先级低于正式接口、评测和提交包整理。

## 基线

- 视觉编码器：复用 `checkpoints_lora_mt/best_model.pt`，CLIP+LoRA 参数冻结。
- 解码器：复用 `checkpoints_bgen/best_decoder.pt`，字符级 GRU。
- 推理：beam=4、长度归一化、数值语法受限解码。
- 当前监督学习基线成绩见 `final_results/bgen_metrics.json`。

## 最小实验

固定种子，在训练集小子集上训练 1～3 轮，在验证集比较监督学习基线。对同一张图，按每步 softmax 分布随机采样一条序列并保留 token 对数概率，同时用 greedy 解码生成一条基线序列。以 `-(R_sample - R_greedy) * sum(log_prob)` 更新解码器，奖励差停止梯度。

奖励定义为加权和：

```text
R = 0.35 * JSON_valid
  + 0.25 * ingredient_F1
  + 0.20 * numeric_score
  + 0.20 * text_similarity
```

其中：

- `JSON_valid`：严格解析且满足 schema 为 1，否则为 0；
- `ingredient_F1`：解析后的食材集合 F1；
- `numeric_score`：五个数值字段的归一化误差得分；
- `text_similarity`：可采用 BLEU-1 或 ROUGE-L 的轻量版本。

使用 self-critical baseline：同一图像下，采样序列的奖励减去 greedy 序列奖励，作为 advantage。仅更新 decoder，保持 CLIP+LoRA 编码器冻结。

## 必须记录的指标

- strict JSON rate、schema validity、fallback rate、empty items rate；
- 食材 micro/macro F1；
- 五个数值字段 MSE、MAE、RMSE；
- BLEU-4、ROUGE-L、CIDEr-D；
- 平均生成长度和平均推理时间；
- reward 均值、greedy 与 sampled 的奖励差。

## 采用标准

只有在 strict JSON rate 不下降超过 1 个百分点，且文本指标或食材 F1 至少一项提升、数值平均 MAE 不恶化超过 3% 时，才考虑保留 RL 权重。否则保留监督学习版本，并在报告中说明 RL 小实验未带来稳定增益。

## 风险与边界

RL 可能偏向固定 JSON 模板，造成文本指标上升但食材和数值变差；也可能破坏数值格式。因此不把 RL 结果作为主提交模型，除非满足上述采用标准。
