import csv,json,platform
from experiment import *
out=ROOT/'results'
notes=[
('蘑菇、碎蒜及绿色香草可见','三种解码均遗漏碎蒜/香草；gravy 不能从图中确认；蘑菇重复'),
('米粒状发酵米、红色枸杞、浅黄色蛋花可见','三种解码把红色配料称辣椒，遗漏米粒/蛋花；RL 的 UNK 重复增加'),
('面包块、红黄番茄、紫洋葱、绿叶可见','Greedy 保留主要食材但重复；Beam/CLIP 丢失主要食材并误称 huevos；CLIP 重复叉子'),
('巧克力表层与碎屑可见，内部不可见','三种解码都称 marshmallow，无可见依据；Greedy/RL 内部白馅无法核实，不计已证实幻觉'),
('煎饼、香蕉、深色酱汁、背景白色饮品可见','Greedy 保留酱汁；Beam/CLIP 重复香蕉，遗漏酱汁；Beam 保留背景饮品'),
('玻璃杯装黄色甜品、白色冰淇淋球可见；具体水果不可辨','CLIP/Beam 的 apple sauce 无可见依据；CLIP 重复冰淇淋；参考的薄荷/柚子在图中无法确认，不按参考硬计遗漏')]
rows=read(out/'reranked.jsonl')[100:106];after={r['file_name']:r for r in read(out/'rl_after.jsonl')}
review=[{'image_id':r['image_id'],'visible_evidence':notes[i][0],'observations':notes[i][1],'greedy':r['greedy'],'beam':r['candidates'][0]['caption'],'clip':r['caption'],'rl':after[r['file_name']]['caption'],'review_type':'assistant visual inspection; first six audit IDs, descriptive not population rate'} for i,r in enumerate(rows)]
dump(out/'visual_review_6.json',review)
r=read(out/'candidates.jsonl');assert len(r)==200 and len({x['image_id'] for x in r})==200
assert all(len(x['candidates'])==4 for x in r)
assert all(c['length']==len(c['token_ids'])-1 and abs(c['generation_score']*c['length']-c['sum_logprob'])<1e-4 for x in r for c in x['candidates'])
rr=json.loads((out/'rerank_metrics.json').read_text(encoding='utf-8'));rl=json.loads((out/'rl_metrics.json').read_text(encoding='utf-8'));ce=json.loads((out/'ce_metrics.json').read_text(encoding='utf-8'))
manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8')); cp=torch.load(DEFAULT_CP,map_location='cpu',weights_only=False)
manifest.update({'checkpoint_epoch':cp.get('epoch'),'config':cp['config'],'python':platform.python_version(),'rl_actual_distinct_train_images':len({r['image_id'] for r in read(out/'rl_history.jsonl')}),'rl_actual_steps':rl['steps']});dump(out/'manifest.json',manifest)
report='''# V4 重排序与 SCST 首轮实测（2026-11-21）

正式 full_v4/best_v4.pt 为共同起点，原权重保持不变。SHA256、配置、固定图片 ID 与随机种子见 results/manifest.json。200 张验证图已生成每图 4 个 Beam 候选，保留 token IDs、含 EOS 的生成长度、累计对数概率、长度归一化分数和图像 ID。Beam width=4，最大长度沿用检查点，无额外去重或重复惩罚。

## 重排序

前 100 张验证图在 alpha=[0,.25,.5,.75,1] 中以本地 CIDEr-D 选系数，后 100 张独立审计。选中 alpha=1；本轮相当于按 CLIP cosine 选择 Beam 候选。分数在每张图片候选内 min-max 归一化，融合为 (1-alpha)*generation+alpha*CLIP。测试集未用于本轮选择。

| 100 张审计验证图 | BLEU-4 (%) | ROUGE-L | CIDEr-D |
|---|---:|---:|---:|
'''
for name,key in [('Greedy','greedy'),('Beam','beam'),('Beam + CLIP','beam_clip')]:
    v=rr['audit'][key];report+=f"| {name} | {v['bleu1_to_4'][3]:.4f} | {v['rouge_l']:.4f} | {v['cider_d']:.4f} |\n"
report+='''
## SCST

ROUGE-L 奖励，loss=-(sample_reward-greedy_reward)*sum(log p(sample))，含 EOS；禁采 PAD/BOS；eval 模式关闭 dropout，sampled 路径保留 autograd，greedy 无梯度。只更新原有可训练参数，Adam lr=1e-6，梯度裁剪 1。固定训练池 64 图，先 32 步，再从包含优化器和随机状态的断点恢复跑第 33 步；实际用了前 33 张训练图，每步一图。此预算只能验证闭环，不能据此否定 RL。CE 对照为同起点未更新模型。

| 相同 200 张验证图 | BLEU-4 (%) | ROUGE-L | CIDEr-D |
|---|---:|---:|---:|
'''
for name,v in [('CE',ce),('SCST',rl['metrics'])]:report+=f"| {name} | {v['bleu1_to_4'][3]:.4f} | {v['rouge_l']:.4f} | {v['cider_d']:.4f} |\n"
report+=f"\n完成 {rl['steps']} 步，非零更新 {rl['nonzero_updates']} 步；平均 sampled 奖励 {rl['mean_sample_reward']:.4f}，greedy 奖励 {rl['mean_greedy_reward']:.4f}。保存权重逐张量核对一致并成功恢复优化器继续更新。完整逐步奖励、梯度范数和参数变化见 rl_history.jsonl。未声称跨设备或中断后逐位复现。\n"
report+='''
## 食材与失败案例

已查看审计集按固定顺序的前 6 张图，详见 visual_review_6.json 和 audit_images.jpg。CLIP 在番茄面包沙拉图上仍选择遗漏主要食材的描述；香蕉煎饼遗漏酱汁；酒酿把枸杞误称辣椒；巧克力派出现没有可见依据的 marshmallow。RL 酒酿案例增加了 UNK 重复。参考描述自身也存在与可见内容不一致的情况，未将参考词表直接当作食材真值。6 张不足以估计整体幻觉/遗漏率；ingredient_review.csv 的其余图仍待人工标注。

## 周末结论

优先继续重排序验证，暂不扩大 RL。下一步先完成这 100 张审计图的食材核查，确认食材误报/遗漏不恶化，再考虑扩大验证集或候选数。保留 Greedy 为默认对照，不替换正式模型。本轮不再追加大型训练。

本表复用项目本地空格分词 BLEU / ROUGE-L / CIDEr-D 实现；BLEU 源码注明训练监控口径，尚未与官方 COCO 工具对齐，不能称官方成绩。CIDEr-D 的文档频率基于各自评价集合，100 图表不能与 200 图表横向直接比较。没有置信区间或显著性结论。测试集历史已被旧报告查看，本轮未重新使用测试集。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
print('Validated 200 x 4 candidates; report generated')

