"""Build the innovation-one report from saved scores, without rerunning model inference."""
import json,csv
from common import ROOT,read,dump

def main():
 t=json.loads((ROOT/'results/coco_metrics.json').read_text(encoding='utf-8'));e=json.loads((ROOT/('results/entity_metrics_human.json' if (ROOT/'results/entity_metrics_human.json').exists() else 'results/entity_metrics.json')).read_text(encoding='utf-8'));m=json.loads((ROOT/'results/manifest.json').read_text(encoding='utf-8'))
 names={'greedy':'Greedy','beam':'Beam','beam_clip':'Beam＋CLIP'}
 report='''# 创新一视觉证据约束候选重排序实验报告

本次完成了V4候选输出、CLIP评分及重排序、冻结配置后的100图审计、文本指标复算和100图助手视觉食材检查，提供独立源码、权重、200张验证图片和复现命令。**工程与首轮实验已完成；开题要求的“人工核对食材实体”尚待组员真实签核。助手标注不能冒充人工标注，本包不宣称全部研究验收已完成。**

当前证据支持继续验证重排序；不支持直接替换正式Greedy模型或宣称稳定降低事实幻觉。本轮不继续盲目扩大实验。

## 方法与划分

原始正式V4 full_v4/best_v4.pt为唯一生成模型起点；CLIP为openai/clip-vit-base-patch32。V4冻结ImageNet ViT-B/32、使用Transformer解码器。原checkpoint的SHA256和配置见results/manifest.json。包内weights/v4_full.pt合入了原冻结骨干，模型参数值未修改；已在3张图验证与此前候选的Greedy、全部Beam token及CLIP分数完全一致。

从原验证集按固定种子20261121选200张，前100张仅用于选择融合系数，后100张为审计集。每张图K=4，保存图像ID、参考、Greedy与候选文本、token ID、含EOS的生成长度、累计生成log概率、平均生成log概率和CLIP余弦相似度。原测试集此前已被历史报告查看，但本轮未读取测试图片或预测，也未用测试集选系数。

Beam扩展按累计log概率截取活动分支，最终候选以sum_logprob/generated_length排序；长度不含BOS，包含实际生成的EOS，未结束时计实际token。PAD和BOS禁止生成，最大长度沿用checkpoint。没有额外重复惩罚。因为搜索过程中按累计分数剪枝，这不是在所有序列上精确求解平均log概率最大值。

候选内分别对生成分数与CLIP余弦做min-max归一化（并列全部取0），S=(1−α)*norm(logP/L)+α*norm(CLIP)。这是开题长度归一化融合的候选内归一化版本，长度指数固定为1，没有搜索长度指数。α网格为0、0.25、0.5、0.75、1；仅在前100张以项目本地CIDEr-D选取，并列选较小α。最终α=1，即本轮最优配置完全按CLIP选取已有候选，不能声称语言分数融合本身获得额外收益。食材检查未再调整α。

## 文本结果

下表使用pycocoevalcap 1.2的BLEU、ROUGE和CIDEr实现，仍明确采用lowercase＋空格分词，不含Java PTB tokenizer，不能称完整官方COCO流水线。逐图CIDEr及统计见results/coco_per_image.json。原项目的200图小样本总表仍保留作历史记录，与100图表不可直接比较CIDEr。

| 后100张验证图 | BLEU-4 % | ROUGE-L | CIDEr-D | 含UNK图片 % | 重复三词图片 % |
|---|---:|---:|---:|---:|---:|
'''
 for key,name in names.items():
  v=t['methods'][key];report+=f"| {name} | {v['bleu_1_to_4_percent'][3]:.4f} | {v['rouge_l']:.4f} | {v['cider_d']:.4f} | {100*v['unk_image_rate']:.1f} | {100*v['repeated_trigram_image_rate']:.1f} |\n"
 ci=t['paired_bootstrap']['beam_clip_minus_greedy']['ci95'];report+=f"\nCLIP相对Greedy的CIDEr-D差值为{t['paired_bootstrap']['beam_clip_minus_greedy']['cider_d_delta']:.4f}，5000次配对图片bootstrap的95%区间为[{ci[0]:.4f}, {ci[1]:.4f}]，跨越零。文档频率保持原100图不变，区间只反映此固定集合下的探索性抽样误差，不能据此声称总体收益显著。CLIP同时增加重复片段，说明语义匹配分数不能保障生成质量。\n"
 report+='''
## 可见食材实体结果

以下全部来自助手视觉检查，不是人工真值。主表使用99张可评图片；编号033疑似聚合物黏土蛋糕模型，是看图后标记的排除。文本评价保留100张，实体结果同时提供保留该图的100图敏感性表，避免隐瞒排除影响。精确率/召回率采用粗粒度可见成分及封闭词典；肉种和香草种类合并，隐蔽配方不推断。无可见证据率是本协议下的视觉支持不足比例，不等于已证明事实错误的严格幻觉率。完整定义、分母和人工复核步骤见ENTITY_PROTOCOL.md。

| 助手标注99图 | 食材精确率 % | 食材召回率 % | 无可见证据率 % | 遗漏率 % |
|---|---:|---:|---:|---:|
'''
 for key,name in names.items():
  v=e['methods'][key];report+=f"| {name} | {v['precision']*100:.2f} | {v['recall']*100:.2f} | {v['unsupported_rate']*100:.2f} | {v['omission_rate']*100:.2f} |\n"
 report+='''
点估计显示CLIP的可见成分精确率与召回率上升，但相对Greedy的配对区间仍跨零。相对Beam的助手指标差值区间未跨零，也不能消除同一助手标注、封闭词典、后验粒度选择和无独立复核的偏差，不能作为严格人工幻觉评测结论。保留全部100图或将无法确认实体按不同边界处理的敏感性结果见entity_metrics.json。

## 典型成功与失败

以下编号对应review拼图及audit_index.json；案例在看结果后选取，不代表发生比例。

- 006：Greedy把蘸酱/玉米片描述成豆与米，Beam及CLIP改为玉米片，纠正主食物方向；CLIP把红酱称ketchup仍缺少配方证据。
- 020：Greedy称卷心菜汤；Beam/CLIP提及可见的鸡肉、面条、胡萝卜和香草，是改善案例。
- 002：Greedy保留面包、番茄及洋葱；Beam/CLIP转而称huevos并重复餐具，主要食材大量遗漏。
- 004：CLIP遗漏可见深色酱汁，重复香蕉；语义相似并不保证覆盖全部食材。
- 070：炸鸡图片被Beam/CLIP称apple crisp，重排序无法修复候选池缺失正确描述的问题。
- 089：可见整只虾，三种方法都误称鸡肉，说明本轮不能解决细粒度食材识别。
- 098：Beam/CLIP比Greedy更接近热狗/香肠主成分，但又添加没有明确依据的洋葱及红椒。
- 033：数据图片疑似非真实食物，应在后续数据审计中核对；本轮未更改正式数据划分或删除原图。

## 可复现与验收状态

已完成：200×4候选和分数、100图选参、100图冻结审计、三个基线、文本实现核对、配对不确定性、逐图助手实体记录、指标脚本、失败案例、全部相关独立源码、原模型及CLIP权重、200张输入图。具体检查见verification/checks.json。没有把强化学习源码、训练权重或日志放入本目录。

待真实人工完成：组员按review/human_review.csv核对100张的可见/无法确认实体并签名日期，运行evaluate_entities.py --human-csv生成独立人工指标，再决定是否可以在报告中使用“人工核对”表述。只有在复核确认改善后，才考虑扩大验证数据；新一轮调参须使用新的独立评估数据。

本轮仍是固定200张验证图的首轮研究，没有新增全量验证或测试集结论；没有加入细粒度检测器、区域证据或实体惩罚。开题将细粒度约束列为增强项，因此无需为了基础版完成再添加一个模型。
'''
 if (ROOT/'review/user_confirmation.json').exists():
  report=report.replace('**工程与首轮实验已完成；开题要求的“人工核对食材实体”尚待组员真实签核。助手标注不能冒充人工标注，本包不宣称全部研究验收已完成。**','**工程与首轮实验已完成。用户于2026年11月21日在聊天中表示已看完且与草稿一致，并授权助手代录100行复核字段。该记录是用户对助手草稿的确认，不是独立人工盲标，也不是助手代签真实姓名。证据见review/user_confirmation.json。**')
  report=report.replace('以下全部来自助手视觉检查，不是人工真值。','以下标注最初来自助手视觉检查，现由用户自述逐图检查并确认草稿不变；助手按授权代录，保留原始标注来源。不能称为独立人工真值。')
  report=report.replace('助手标注99图','用户确认草稿99图')
  report=report.replace('无独立复核的偏差','无独立盲标的偏差')
  report=report.replace('待真实人工完成：组员按review/human_review.csv核对100张的可见/无法确认实体并签名日期，运行evaluate_entities.py --human-csv生成独立人工指标，再决定是否可以在报告中使用“人工核对”表述。只有在复核确认改善后，才考虑扩大验证数据；新一轮调参须使用新的独立评估数据。','用户确认记录已完成：review/human_review.csv的100行已按用户确认代录，human_visible/human_uncertain与草稿一致，99图可评、编号033仍排除。已重算entity_metrics_human.json，数值与草稿一致。reviewer记录为“用户（本聊天确认，助手代录）”，未虚构真实姓名或电子签名。该记录可以表述为“助手辅助标注，经用户检查确认”，不能表述为独立双人盲审。基础版可据此整理收尾；更强的研究结论仍建议独立复核和更大评估集。')
  report=report.replace('人工指标完成后应相应更新最终报告','用户确认指标已更新最终报告')
 (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
 rows=read(ROOT/'results/reranked.jsonl')[100:]
 with (ROOT/'review/captions_and_entities.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['index','image_id','reference','greedy','beam','beam_clip'])
  for i,r in enumerate(rows):w.writerow([i,r['image_id'],r['reference'],r['greedy'],r['candidates'][0]['caption'],r['caption']])
 print('Innovation-one report generated')
if __name__=='__main__':main()
