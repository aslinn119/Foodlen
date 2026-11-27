# FoodLens 抽样复核（5 图 / 20 条）分歧分析

日期：2026-11-16（Asia/Shanghai）
范围：`outputs/judge_four_model/deepseek_web_formal`（30 组图片 × 4 条候选）中的 5 组。

## 0. 交付物

| 文件 | 说明 |
|---|---|
| `human_review_visual_1to5_blank.csv` | **已回填 20 行**（本次抽中的 5 组），其余 100 行保持空白 |
| `human_review_visual_1to5_blank.backup.csv` | 回填前的原件备份 |
| `human_review_sample5/selection.md` | 选图记录（随机种子、3 张随机图、2 张问题图及理由） |
| `human_review_sample5/raw/NN_review.json` | 5 位复核员的原始 JSON（含四条候选的分数、看图依据、无法确认项、疑似幻觉） |
| `human_review_sample5/sample5_review_table.csv` | 20 行紧凑对照表（复核 vs DeepSeek） |
| `deepseek_web_formal/results/judge_scores.jsonl` | DeepSeek 正式评分（未改动） |

## 1. 选图

**3 张随机图：08、14、23**（`random.Random(20261116).sample(01..30 去掉 {03,19}, 3)`，种子可复现）；
**2 张问题图：03、19**——
- 03：词表假阳性案例，03A 写 "fish steak"，Task2 词表按 "steak" 与 Task2 输出判为共有食材；
- 19：Task2 明显过度预测案例，对墨鱼汁黑米饭输出 13 项蔬菜，且 19B「red cabbage」与 Task2「cabbage」再次词面误配。

详见 `selection.md`。

## 2. 复评方法与限制

- 每张图由**一个独立子 Agent**完成：先 `read_image` 看原图，再读该图 A–D 四条候选；**未提供** DeepSeek 分数、候选模型身份、其他图片组内容。
- 四个维度 1–5 整数，锚点沿用项目正式提示词；要求逐条给出看图依据、无法确认项、明显遗漏、疑似幻觉。
- **限制（务必注意）**：复核员是 AI 子 Agent（DeepSeek V4.1 Flash，Max，同一模型的不同上下文），**不是真人**。本表适合做复现性检验与案例初筛；报告若写"人工复核"，请由真人重新签认后再定稿。`reviewer` 列已如实写明"非真人"。

## 3. 一致性总览

- 20 条 × 4 维 = **80 个分数对**：完全一致 **54**（67.5%）；相差 ≤1 分 **78**（97.5%）；**相差 ≥2 分仅 2 处**。
- 各维度平均绝对差：视觉 0.35、识别 0.45、细节 0.2、语言 0.4。

| 编号 | 候选描述（截断） | DeepSeek 视觉/识别/细节/语言 | 复核 视觉/识别/细节/语言 | 分差≥2 或判断差异 |
|---|---|---|---|---|
| 08A | a plate of stuffed mushrooms with chopped green onion and minced… | 4/4/3/5 | 4/5/3/5 | — |
| 08B | a brown filet mignon with a crispy surface with a crispy surface… | 2/1/2/3 | 2/1/2/2 | — |
| 08C | a plate of stuffed eggplants with chopped green onion and choppe… | 2/1/2/3 | 3/2/2/2 | — |
| 08D | a plate of stuffed mushrooms with diced tomatoes, and chopped pa… | 3/4/2/5 | 4/5/2/4 | diced tomatoes 定性不同：无法确认 vs 疑似幻觉 |
| 14A | a plate of brown stew chicken with chopped green onion and red p… | 3/5/3/5 | 3/5/3/5 | red pepper 是否算错：复核列为无依据 |
| 14B | a pot of brown stew peas with sliced beef and green beans | 3/2/2/4 | 4/2/2/5 | — |
| 14C | a plate of beef chow fun with chopped green onion, and a few pie… | 1/1/2/4 | 3/2/2/4 | 视觉差2分（DS 1 / 复核 3） |
| 14D | a pot of chicken marsala with chopped spring onion and chopped s… | 4/3/3/2 | 4/3/4/3 | — |
| 23A | a plate of dark brown colored pipikaula with a drizzle of soy sa… | 3/2/2/4 | 2/1/2/4 | — |
| 23B | a plate of brown pig's ear served with a small bowl of sauce on … | 4/5/3/5 | 4/5/3/5 | — |
| 23C | a plate of <UNK> with a <UNK> of <UNK> of <UNK> <UNK> <UNK> <UNK… | 2/1/1/1 | 2/1/1/1 | — |
| 23D | a plate of dark brown torta caprese with a drizzle of soy sauce … | 2/1/2/5 | 2/1/2/5 | — |
| 03A | a large piece of fish steak with a super crispy surface and a fe… | 1/1/1/4 | 2/1/1/4 | — |
| 03B | a large piece of <UNK> raw meatloaves with <UNK> <UNK> and green… | 3/3/3/2 | 3/3/2/1 | — |
| 03C | a large piece of porchetta with a slightly charred skin and some… | 2/2/2/5 | 2/2/2/5 | — |
| 03D | a loaf of freshly baked bacon explosion with a glossy surface an… | 2/1/2/3 | 2/2/3/4 | — |
| 19A | a plate of black arros negre with a whole red pepper on top | 4/5/3/4 | 4/3/3/4 | 识别差2分（DS 5 / 复核 3） |
| 19B | a plate of red slaw with shredded red cabbage, red cabbage, red … | 1/1/1/2 | 1/1/1/2 | — |
| 19C | a plate of black arros negre with a lot of chili oil and some ch… | 3/4/2/5 | 3/3/2/4 | chili oil 定性不同：冲突 vs 幻觉 |
| 19D | a plate of black arros negre with a lot of cuttlefish and a few … | 3/3/2/4 | 3/3/3/4 | — |

## 4. 两处 ≥2 分分歧

### 4.1 14C · 视觉一致性 DeepSeek 1 / 复核 3（差 2）

14C：DeepSeek 1/1/2/4；复核 3/2/2/4；候选原文「a plate of beef chow fun with chopped green onion, and a few pieces of beef on the top」

- DeepSeek：容器写成 "a plate of"，图中是不锈钢深锅，直接判 1。
- 复核：同样认为"盘子"是误判，但认为"锅内一堆块状食物 + 确有切碎葱"的整体构图仍部分对应，给 3。
- 性质：不是事实分歧，而是**对"容器错误"扣分力度不同**；建议报告里把该条视觉分数标注为"低分档（1–3）"，不把具体分值当硬结论。

### 4.2 19A · 食物识别 DeepSeek 5 / 复核 3（差 2）

19A：DeepSeek 4/5/3/4；复核 4/3/3/4；候选原文「a plate of black arros negre with a whole red pepper on top」

- 两边都认出主体是 black arros negre。
- DeepSeek 只看主类别正确即给 5；复核认为饭上最显眼的整只虾与青口贝完全没写，食物信息不完整，只给 3。
- 性质：**"识别准确"是否应包含主要配料完整性**的口径差异，提示"食物识别准确性"这一维度需要在 rubric 里写明是否含配料。

## 5. 三处"是否写错食材"判断口径差异（分值差 ≤1）

| 编号 | 争议点 | DeepSeek | 复核 | 处理建议 |
|---|---|---|---|---|
| 14A | "red pepper" | 只列容器冲突，未判红椒错 | 列为疑似无依据（图中椒片为青绿、无红椒） | 记"配料存疑"，不升级为幻觉 |
| 08D | "diced tomatoes" | 列为疑似幻觉 | 列为"无法确认"（仅一个极小红色颗粒） | 保留"无法确认"，不判幻觉 |
| 19C | "a lot of chili oil" | 归"与画面冲突" | 归"疑似幻觉" | 统一口径：画面不可见且无合理依据 → 记"疑似幻觉" |

## 6. Task2 同图对照（两边都提到 / 谁漏写 / 无法判断）

> Task2 = 模型 B（CLIP+LoRA）对同一张 30 图清单重新推理的营养 JSON；词表匹配结果见 `task2_same_image/ingredient_comparison.jsonl`。词面重合**不能**单独作为"食材一致"的证据。

### 08（酿蘑菇）
- **两边都提到**：几乎没有共享词。08A/08D 写了 mushroom（正确主体），Task2 未输出 mushroom；Task2 的 parmesan cheese 与画面顶部奶酪相近，但四条候选都没写 cheese。
- **漏写**：候选侧漏奶酪层；Task2 侧漏蘑菇主体、肉末、奶酪，反而输出 blueberries/carrot/chicken/spinach。
- **无法判断**：绿色碎末是香草还是葱；红色小点是否番茄；奶酪种类；是否放在盘上。

### 14（棕色炖鸡）
- **两边都提到**：鸡肉（14A、14D 与 Task2 共有 "chicken"）。
- **漏写**：Task2 输出 almonds/bell peppers/carrot/country rice/spinach/wheat berry，画面均无对应；14B/14C 把主料写成 beef；四条候选普遍漏浅色蒜/洋葱块与青椒片。
- **无法判断**：绿色椒片品种；酱汁成分；带皮鸡块部位；Task2 的 carrot/bell peppers 是否对应图中浅色块。

### 23（卤猪耳三格蘸酱）
- **两边都提到**：无共享词。23B 正确写 pig's ear，Task2 却输出 bacon/bok choy，词表无交集。
- **漏写**：Task2 漏猪耳与三格蘸酱；候选普遍漏三格蘸酱（23B 简化成"一只小碗"）与绿叶；23C 全为 `<UNK>`。
- **无法判断**：三格蘸酱成分；"酱油"是否准确；绿叶品种。

### 03（烤网上的腌制牛肋排）
- **两边都提到**：仅词面 "steak"（03A "fish steak" ↔ Task2 "steak"）。03A 的 "fish" 本身错误，Task2 那边是 beef/steak，**词面重合但语义不一致**，不能算一致。
- **漏写**：Task2 漏黑色烤网、腌料、带骨牛肋排等显著元素，还输出 salmon/bok choy/cherry tomatoes 等无据项；候选普遍漏两块数量、菱形刀花、白色脂肪边。
- **无法判断**：具体部位（牛肋排/牛仔骨）；腌料成分；绿叶品种。

### 19（墨鱼汁黑米饭）
- **两边都提到**：仅词面 "cabbage"（19B "red cabbage" ↔ Task2 "cabbage"）；19B 整条描述是红甘蓝沙拉，画面并无甘蓝，**两边都错**。真实主体（黑米饭、虾、青口贝）两边都没可靠写到。
- **漏写**：Task2 完全没输出米饭/虾/贝，只给 13 项绿叶蔬菜；19A/19C/19D 写了黑米饭但漏整只虾与青口贝；19B 全错。
- **无法判断**：是否墨鱼汁调味；红椒品种；饭粒间绿色碎点是否葱花。

## 7. 结论（可直接写进报告的一段）

> 抽样复核采取"3 张随机图 + 2 张问题图、每图 A–D 四条"的设计，共 20 条。以独立子 Agent 盲评复现 DeepSeek 评分：80 个分数对中 54 个完全一致（67.5%），78 个相差不超过 1 分（97.5%），仅 2 处相差 ≥2 分（14C 视觉一致性 1 vs 3、19A 食物识别 5 vs 3），其余差异集中在"容器/配料是否算错"的定性口径上。据此可判断正式评分整体可复现，但"食物识别准确性"维度需明确是否包含配料完整性，"与画面冲突/无法确认/疑似幻觉"三类标签需统一口径。Task2 同图对照进一步显示：自动词表匹配存在假阳性（03A 的 "fish steak" 被算作与 "steak" 一致、19B 的 "red cabbage" 与 Task2 的 "cabbage" 误配），Task2 自身也会过度预测（19 号图输出 13 项画面并不存在的绿叶蔬菜）。因此 Task2 词表指标与 DeepSeek 自动评分都只能作为复核线索，不能单独作为营养精度或幻觉率的结论依据，最终结论需结合人工看图。

## 8. 说明与待办

- 20 条的复核分数与看图依据已写入 `human_review_visual_1to5_blank.csv`；`judge_disagreement` 列只填了存在分歧的 5 条，其余标注"无（四项差均≤1分，食材判断一致）"。
- 其余 100 条未复核，空表结构未改（字段、顺序、file_name/reference/candidate 原文均保持）。
- 若需要真人在表上签认：可保留本表作为预填，在 `reviewer` 列改为真人姓名/日期，并按需调整 `judge_disagreement`。
