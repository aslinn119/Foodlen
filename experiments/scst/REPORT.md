# V4 SCST 强化学习正式预算实验报告

本实验按开题4.4完成SCST训练、CE对照、文本指标、训练稳定性与视觉食材变化评价。采用固定预算和单随机种子，不以提升为验收条件，不声称复现论文的大规模训练效果。

## 实验设计

以正式V4 full_v4交叉熵检查点为共同起点，保留冻结ViT-B/32。固定抽取2048张训练图，batch size 8、3轮，每条训练图每个模型看到3次；每个训练臂768次Adam更新，lr=1e-6、梯度裁剪1。训练投影层和解码器，冻结编码器；缓存未经投影的float32视觉特征，投影层始终保留梯度。两臂均关闭dropout，避免采样策略与打分策略不一致。

SCST使用CIDEr-D奖励：loss = -mean[(R(sample)-R(greedy))*sum(log p(sample))]。采样禁用PAD/BOS，概率求和包含EOS，EOS后PAD不计分。采样无梯度，随后对同一采样序列进行可微分teacher forcing打分；与逐词打分的一致性检查见implementation_checks.json。CIDEr训练奖励的文档频率只由全部训练参考建立，禁止逐minibatch或测试数据重估。

另设原CE冻结模型和同预算继续CE训练模型。后者不是SCST本身，而是排查额外训练步数影响的控制。对照保持数据次序、更新次数、学习率和模型起点一致；目标函数数值尺度及计算量并不相同。只有一个种子，不声称跨种子稳定或统计上普遍优于。

256张固定验证图用于逐轮选择最高CIDEr检查点，并列取最早轮；与100张视觉审计图严格分离。测试集4938图仅在每个训练臂检查点选定后统一评价，不用来改超参。项目历史上已查看测试集，本轮只能称回溯评估，不称全新未见测试基准。文件名、图片字节重复和缺图检查见data_audit.json；该检查不保证不存在近似重复。

## 指标协议

采用pycocoevalcap的BLEU、ROUGE-L和CIDEr实现，统一小写及空格分词，不使用PTB/Java，不能称完整官方COCO流水线。BLEU乘100，ROUGE-L为0—1，CIDEr为x10实现的原数值。测试CIDEr文档频率由同一测试参考集合建立，各模型保持一致。训练奖励和测试分数因DF语料不同不可直接比高低。

## 测试集结果

| 模型 | 样本数 | BLEU-4 % | ROUGE-L | CIDEr-D | UNK图片率 % | 重复三词图片率 % |
|---|---:|---:|---:|---:|---:|---:|
| ce_original | 4938 | 4.7523 | 0.2557 | 0.3242 | 18.73 | 25.50 |
| ce_continued | 4938 | 4.8110 | 0.2598 | 0.3345 | 14.74 | 18.93 |
| scst | 4938 | 4.7879 | 0.2563 | 0.3244 | 19.08 | 25.64 |

## 稳定性与检查点

- ce_continued：完成768次更新，非零更新768次；最优为第3轮，选择集CIDEr-D=0.3367。所有loss有限，保存张量逐项一致性检查通过。
- scst：完成768次更新，非零更新768次；最优为第1轮，选择集CIDEr-D=0.3197。所有loss有限，保存张量逐项一致性检查通过。

SCST梯度裁剪前范数均值51.7716，最大185.1814。奖励与梯度曲线见[训练与结果图](results/training_and_results.png)，完整逐步记录见scst_history.jsonl。保存了模型、优化器、CPU/CUDA及Python随机状态；支持轮边界恢复，未宣称任意中断或跨设备逐位复现。

触发裁剪的更新比例为100.00%；采样奖励均值0.1550，贪心奖励均值0.4289。有限loss和梯度表明本次未数值发散，不等于已经证明跨种子训练稳定。

## 文本收益的不确定性

- scst_minus_ce_original：CIDEr-D差值0.0002，配对图片bootstrap 95%区间[-0.0031, 0.0035]。
- scst_minus_ce_continued：CIDEr-D差值-0.0102，配对图片bootstrap 95%区间[-0.0194, -0.0011]。

该区间固定参考文档频率，只描述此测试集合的图片抽样误差，不包含训练种子差异或历史测试使用偏差。

## 视觉食材审计

复用创新一100张图的既有标注：来源是助手辅助标注、用户确认，非独立人工盲标。保留原编号033疑似非真实食物图片的排除，主表99图，另给全部100图敏感性结果。粗粒度封闭词表、肉种和香草合并；不确定实体中立。无可见证据不等于已证明事实错误，不以参考caption自动构造视觉真值。标注未用于奖励或检查点选择，但历史上已观察过，结论属探索性。

| 模型 | 食材精确率 % | 食材召回率 % | 无可见证据率 % | 遗漏率 % |
|---|---:|---:|---:|---:|
| ce_original | 42.92 | 34.34 | 57.08 | 65.66 |
| ce_continued | 45.96 | 34.34 | 54.04 | 65.66 |
| scst | 43.66 | 35.09 | 56.34 | 64.91 |

逐图结果、5000次配对bootstrap及100图敏感性见entity_metrics.json和entity_per_image.jsonl。cases.jsonl按“无证据实体数+遗漏实体数”的减少值选择最大改善3例和最大退化3例，为看结果后选例，不能用6例代表总体发生率。

## 典型案例

### val/Mongolian_beef/Mongolian_beef_00037.jpg

本协议下实体错误数减少2，选择规则：largest_entity_improvement。

- ce_original：a plate of andong jjimdak with braised chicken, carrot and green pepper；无可见证据实体：carrot, pepper；遗漏：onion, scallion。
- ce_continued：a plate of andong jjimdak with braised chicken, carrots, and cellophane noodles, topped with green onions.；无可见证据实体：carrot, noodle；遗漏：onion。
- scst：a plate of andong jjimdak with braised chicken, carrots, and green onions；无可见证据实体：carrot；遗漏：onion。

### val/Rice_and_peas/Rice_and_peas_00036.jpg

本协议下实体错误数减少2，选择规则：largest_entity_improvement。

- ce_original：a red bowl of steamed rice with green beans and white rice；无可见证据实体：bean；遗漏：pea。
- ce_continued：a red bowl of colcannon with mashed potatoes and chopped green onions on top；无可见证据实体：potato, scallion；遗漏：pea, rice。
- scst：a red bowl of steamed rice with green peas and white rice on the surface.；无可见证据实体：无；遗漏：无。

### val/Crab_dip/Crab_dip_00036.jpg

本协议下实体错误数减少1，选择规则：largest_entity_improvement。

- ce_original：a bowl of yellow and white rice pudding with a spoon on the side；无可见证据实体：rice；遗漏：breadcrumb。
- ce_continued：a bowl of yellow and white grits porridge sprinkled with some bacon crumbs.；无可见证据实体：bacon；遗漏：breadcrumb。
- scst：a bowl of yellow and white grits with a spoon on it.；无可见证据实体：无；遗漏：breadcrumb。

### val/Turnip_cake/Turnip_cake_00037.jpg

本协议下实体错误数减少-1，选择规则：largest_entity_regression。

- ce_original：a piece of apple strudel with a brown crust on the top and a white plate of <UNK> beside it.；无可见证据实体：apple；遗漏：无。
- ce_continued：a piece of turnip cake with a brown surface and a fork on a white round plate.；无可见证据实体：无；遗漏：无。
- scst：a piece of apple strudel with a brown crust on the top and a white plate of ice cream on the side.；无可见证据实体：apple, ice_cream；遗漏：无。

### val/Fried_chicken/Fried_chicken_00038.jpg

本协议下实体错误数减少-1，选择规则：largest_entity_regression。

- ce_original：a plate of golden crisp fried chicken with a crisp crust, and a little bit of <UNK>；无可见证据实体：无；遗漏：无。
- ce_continued：a pile of golden brown <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> on a white plate；无可见证据实体：无；遗漏：meat。
- scst：a plate of golden crisp fried chicken with a crisp crust, and some white cheese on the surface.；无可见证据实体：cheese；遗漏：无。

### val/Cabbage_soup/Cabbage_soup_00038.jpg

本协议下实体错误数减少-1，选择规则：largest_entity_regression。

- ce_original：a pot of tomato bredie made with diced tomatoes, potatoes, and tomatoes, served with a spoon on the side；无可见证据实体：无；遗漏：cabbage, carrot。
- ce_continued：a pot of tomato bredie made with tomatoes, potatoes, and tomatoes, sprinkled with some black pepper.；无可见证据实体：无；遗漏：cabbage, carrot。
- scst：a pot of tomato bredie made with tomatoes, potatoes, and tomatoes, sprinkled with some chopped green onions；无可见证据实体：scallion；遗漏：cabbage, carrot。

## 结论与局限

本轮SCST相对原CE的测试CIDEr-D点估计上升0.0002，无可见证据率点估计下降0.74个百分点。数值奖励优化不自动保证视觉事实改善，需同时参考完整文本指标、重复率、实体指标和区间。

本轮完成固定预算的最小完整实验，训练规模为2048图×3轮，不能写成全量训练或多种子稳定收益。无需因负结果继续在测试集调参。更大训练预算、多种子、独立真人盲标属于后续增强，不混入本轮结论。大权重与特征缓存不提交普通Git，检查点哈希和原始输入清单保留以追溯。

方法依据：[Rennie等，CVPR 2017，Self-Critical Sequence Training](https://openaccess.thecvf.com/content_cvpr_2017/html/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.html)。

## 额外看图核查

助手在冻结计分后另查看了米饭豌豆、炸鸡、萝卜糕三例。米饭例的豌豆修正有可见支持；炸鸡例新增奶酪、萝卜糕例新增冰淇淋均缺少明确视觉支持。炸鸡输入本身是拼接图，记录为数据局限，未为改善指标后验剔除。此检查未修改标注或训练策略，也不构成独立真人复评。详见results/assistant_visual_spotcheck.json。

## 本轮比较结论

SCST对原CE的CIDEr差异区间跨零，未观察到可靠文本收益；在此固定集合下低于同预算继续CE。实体误报的配对区间也跨零，不能声称稳定抑制幻觉。保留原正式模型身份，不把SCST自动设为系统默认。
