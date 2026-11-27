# FoodLens 正式盲评交付报告（deepseek_web_formal）

生成时间：2026-11-16（Asia/Shanghai）
材料目录：`阶段2_统一评测与可信分析/outputs/judge_four_model/deepseek_web_formal`
执行方式：DSH 子 Agent 实际看图评分（非主 Agent 代评）

## 1. 模型与设置（全程一致）

| 项目 | 值 |
|---|---|
| 提供方 | OpenCode Go（provider: `cc-switch-open-code-go`） |
| 模型 | DeepSeek V4.1 Flash 1017（model: `deepseek-v4.1-flash`） |
| 推理强度 | Max（`reasoning_effort: max`） |
| 界面 | DSH 子 Agent（每条候选一个独立子 Agent） |
| 提示词版本 | 正式目录内各 `NN<A-D>.txt` 全文（`formal_web_v1`） |
| 并行上限 | 同时最多 4 个子 Agent；首组 01A–01D 四路同时启动 |

## 2. 执行与隔离

- 每条候选由**独立子 Agent**完成：先 `read_image` 读取该组 `image.jpg`（真实图块，不是路径或文字描述），再 `read` 读取该候选的 `.txt` 全文。
- 未向子 Agent 提供：其他候选、已有评分、模型身份、主 Agent 对图片的任何描述。
- 子 Agent 被限制为只读上述两个文件路径，禁止浏览同组其他文件。
- 首组 01A–01D 验收：四条均返回与像素一致的画面细节（含图片左上角水印、白碗边缘等仅像素可见信息），确认实际看图；`candidate_id` 全部正确；四项分数均为 1–5 整数；`reasons` 四项齐全；四个问题字段均为字符串数组。验收通过后才扩展到 02–30。
- 全部 120 条均无 `image_unavailable`。

## 3. 结果数量

| 指标 | 数量 |
|---|---|
| 预期 | 120 |
| 有效结果 | **120** |
| 失败 / 缺失 | **0** |
| `image_unavailable` | 0 |
| 覆盖图片 | 30 组（01–30），每组 4 条 |

格式重试记录：**05B** 第 1 次回复的 `reasons` 写成了 `{score, reason}` 嵌套对象（分数合法但结构不合格），按规则用同一原图、同一提示词、同一模型重试 1 次后通过（2/2/3/5），以重试结果为最终结果。

## 4. 评分分布（120 条）

| 维度 | 1 | 2 | 3 | 4 | 5 | null |
|---|---:|---:|---:|---:|---:|---:|
| visual_consistency | 16 | 40 | 39 | 24 | 1 | 0 |
| food_identification | 47 | 26 | 12 | 22 | 13 | 0 |
| detail_accuracy | 22 | 72 | 20 | 6 | 0 | 0 |
| language_quality | 5 | 26 | 17 | 43 | 29 | 0 |

四个问题分类字段的非空条数：`explicit_conflicts` 102、`unverifiable_from_image` 79、`visible_but_omitted` 119、`suspected_ingredient_hallucinations` 80。

## 5. 结果路径

均在 `deepseek_web_formal/results/` 下：

| 文件 | 说明 |
|---|---|
| `judge_scores.jsonl` | **汇总交付**：120 行，每行一个候选编号，含分数、理由、四个问题数组、参考/候选文本与评判模型信息 |
| `01A.json` … `30D.json` | 逐条结果，共 120 个，内容为模型返回的 JSON 对象 |
| `raw/01A.txt` … `raw/30D.txt` | 模型原始回复，共 120 个（纯 JSON，未改写） |
| `manifest.json` | 运行设置、逐条状态、重试记录、汇总 |
| `logs/batches.txt` | 分批执行日志与每批看图核对记录 |
| `_invalid.json` | 校验失败清单（当前为空 `[]`） |
| `_tools/build_results.py` | 校验/汇总脚本，可复现 `judge_scores.jsonl` |

## 6. 说明

- 原始模型评分按原样保留，未做人工修正；`judge_scores.jsonl` 中 `human_review` 字段统一为 `null`，供后续人工复核另填。
- 人工复核表仍在上级目录：`human_review_blank.csv`、`human_review_visual_1to5_blank.csv`（保持空白未改动）。
- 此前试跑记录（`deepseek_web_pilot/`）未改动。
- 采样温度/参数在桌面 UI 与子 Agent 接口中均无显式设置；已记录实际模型、提供方与推理强度。
