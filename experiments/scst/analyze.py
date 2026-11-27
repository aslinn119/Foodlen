"""Summarize actual run outputs, paired uncertainty, and frozen visible-entity audit."""
import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np
from entities import extract
from run import read,dump,lines

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results');a=p.parse_args();out=a.out
    methods=['ce_original','ce_continued','scst'];annotations=read(out/'entity_annotations.jsonl')
    rows={m:{r['image_id']:r for r in read(out/f'{m}_entity_audit_predictions.jsonl')} for m in methods}
    per=[];aggregate={}
    for m in methods:
        for ann in annotations:
            r=rows[m][ann['image_id']];pred=set(extract(r['caption']));visible=set(ann['visible']);uncertain=set(ann['uncertain'])-visible
            per.append({'image_id':ann['image_id'],'method':m,'caption':r['caption'],'eligible':ann['eligible'],
              'predicted':sorted(pred),'matched':sorted(pred&visible),'unsupported':sorted(pred-visible-uncertain),
              'omitted':sorted(visible-pred),'unresolved':sorted(pred&uncertain),'visible':sorted(visible),'uncertain':sorted(uncertain)})
    def entity_metrics(rr):
        tp=sum(len(r['matched']) for r in rr);fp=sum(len(r['unsupported']) for r in rr);fn=sum(len(r['omitted']) for r in rr)
        return {'tp':tp,'unsupported':fp,'omitted':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
          'unsupported_rate':fp/(tp+fp) if tp+fp else None,'omission_rate':fn/(tp+fn) if tp+fn else None,
          'unsupported_image_rate':sum(bool(r['unsupported']) for r in rr)/len(rr),'zero_entity_captions':sum(not r['predicted'] for r in rr)}
    for m in methods:
        aggregate[m]=entity_metrics([r for r in per if r['method']==m and r['eligible']])
    lines(out/'entity_per_image.jsonl',per)
    sensitivity={m:entity_metrics([r for r in per if r['method']==m]) for m in methods}
    rng=np.random.default_rng(20261127);n=sum(a['eligible'] for a in annotations);samples=rng.integers(0,n,(5000,n))
    boots={}
    for m in methods:
        rr=[r for r in per if r['method']==m and r['eligible']]
        tp=np.array([len(r['matched']) for r in rr])[samples].sum(1)
        fp=np.array([len(r['unsupported']) for r in rr])[samples].sum(1)
        fn=np.array([len(r['omitted']) for r in rr])[samples].sum(1)
        boots[m]={'precision':tp/np.maximum(1,tp+fp),'recall':tp/np.maximum(1,tp+fn),'unsupported_rate':fp/np.maximum(1,tp+fp)}
    ci={}
    for base in ['ce_original','ce_continued']:
        ci['scst_minus_'+base]={k:{'delta':aggregate['scst'][k]-aggregate[base][k],
          'ci95':np.quantile(boots['scst'][k]-boots[base][k],[.025,.975]).tolist()} for k in boots[base]}
    dump(out/'entity_metrics.json',{'eligible_images':n,'total_images':100,'annotation_source':'reused assistant-assisted annotations confirmed by user; not independent blind human annotation',
      'interpretation':'unsupported means no visible evidence under a coarse closed lexicon; not proved factual absence',
      'methods':aggregate,'paired_bootstrap':ci,'sensitivity_all100':sensitivity,'uncertain_entities':'neutral, excluded from assessable precision denominator'})
    test={m:read(out/f'{m}_test_predictions.jsonl') for m in methods}
    assert all([r['image_id'] for r in test[m]]==[r['image_id'] for r in test['ce_original']] for m in methods)
    test_ci={}
    for base in ['ce_original','ce_continued']:
        d=np.array([s['cider_d']-b['cider_d'] for s,b in zip(test['scst'],test[base])]);vals=[]
        for _ in range(5000):vals.append(float(d[rng.integers(0,len(d),len(d))].mean()))
        test_ci['scst_minus_'+base]={'delta':float(d.mean()),'ci95':np.quantile(vals,[.025,.975]).tolist(),'n':len(d),
          'note':'paired image bootstrap; DF held fixed; single training seed, not training-seed uncertainty'}
    dump(out/'test_paired_bootstrap.json',test_ci)
    # Case choice is explicit, post-hoc, and never used to tune training.
    grouped={r['image_id']:{x['method']:x for x in per if x['image_id']==r['image_id']} for r in per if r['eligible']}
    def improvement(g):return len(g['ce_original']['unsupported'])+len(g['ce_original']['omitted'])-len(g['scst']['unsupported'])-len(g['scst']['omitted'])
    sorted_cases=sorted(grouped.items(),key=lambda kv:(-improvement(kv[1]),kv[0]))
    selected=[]
    for label,pool in [('largest_entity_improvement',sorted_cases[:3]),('largest_entity_regression',list(reversed(sorted_cases))[:3])]:
        for ident,g in pool:selected.append({'selection':label,'image_id':ident,'entity_error_count_reduction':improvement(g),'methods':g})
    lines(out/'cases.jsonl',selected)
    results={m:json.loads((out/f'{m}_test_metrics.json').read_text()) for m in methods}
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['model','n','BLEU4_percent','ROUGE_L','CIDEr_D','UNK_image_rate','repeated_trigram_image_rate','entity_precision','entity_recall','unsupported_rate'])
        for m in methods:
            v=results[m];e=aggregate[m];w.writerow([m,v['n'],v['bleu_1_to_4_percent'][3],v['rouge_l'],v['cider_d'],v['unk_image_rate'],v['repeated_trigram_image_rate'],e['precision'],e['recall'],e['unsupported_rate']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,7))
    hist=read(out/'scst_history.jsonl');xs=[r['step'] for r in hist]
    def smooth(key):return np.convolve([r[key] for r in hist],np.ones(32)/32,mode='valid')
    for key in ['sample_reward','greedy_reward']:ax[0,0].plot(xs[31:],smooth(key),label=key)
    ax[0,0].set_title('Training CIDEr-D reward (32-step moving mean)');ax[0,0].legend()
    ax[0,1].plot(xs,[r['grad_norm'] for r in hist]);ax[0,1].axhline(1,color='red',linestyle='--');ax[0,1].set_title('Gradient norm before clipping (clip=1)')
    for m in ['ce_continued','scst']:
        ep=json.loads((out/f'{m}_epochs.json').read_text());ax[1,0].plot([r['epoch'] for r in ep],[r['metrics']['cider_d'] for r in ep],marker='o',label=m)
    ax[1,0].set_title('Checkpoint selection: fixed 256 validation images');ax[1,0].legend()
    ax[1,1].bar(methods,[results[m]['cider_d'] for m in methods]);ax[1,1].set_title('Final test CIDEr-D (4938 images)')
    for x in ax.flat:x.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'training_and_results.png',dpi=160);plt.close(fig)
    protocol=json.loads((out/'protocol.json').read_text());completions={m:json.loads((out/f'{m}_completion.json').read_text()) for m in ['ce_continued','scst']}
    text='''# V4 SCST 强化学习正式预算实验报告

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
'''
    for m,v in results.items():text+=f"| {m} | {v['n']} | {v['bleu_1_to_4_percent'][3]:.4f} | {v['rouge_l']:.4f} | {v['cider_d']:.4f} | {v['unk_image_rate']*100:.2f} | {v['repeated_trigram_image_rate']*100:.2f} |\n"
    text+='\n## 稳定性与检查点\n\n'
    for m,v in completions.items():text+=f"- {m}：完成{v['steps']}次更新，非零更新{v['nonzero_updates']}次；最优为第{v['best_epoch']}轮，选择集CIDEr-D={v['best_validation_cider_d']:.4f}。所有loss有限，保存张量逐项一致性检查通过。\n"
    text+=f"\nSCST梯度裁剪前范数均值{np.mean([r['grad_norm'] for r in hist]):.4f}，最大{max(r['grad_norm'] for r in hist):.4f}。奖励与梯度曲线见[训练与结果图](results/training_and_results.png)，完整逐步记录见scst_history.jsonl。保存了模型、优化器、CPU/CUDA及Python随机状态；支持轮边界恢复，未宣称任意中断或跨设备逐位复现。\n"
    text+=f"\n触发裁剪的更新比例为{np.mean([r['grad_norm']>1 for r in hist])*100:.2f}%；采样奖励均值{np.mean([r['sample_reward'] for r in hist]):.4f}，贪心奖励均值{np.mean([r['greedy_reward'] for r in hist]):.4f}。有限loss和梯度表明本次未数值发散，不等于已经证明跨种子训练稳定。\n"
    text+='\n## 文本收益的不确定性\n\n'
    for k,v in test_ci.items():text+=f"- {k}：CIDEr-D差值{v['delta']:.4f}，配对图片bootstrap 95%区间[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]。\n"
    text+='\n该区间固定参考文档频率，只描述此测试集合的图片抽样误差，不包含训练种子差异或历史测试使用偏差。\n\n## 视觉食材审计\n\n复用创新一100张图的既有标注：来源是助手辅助标注、用户确认，非独立人工盲标。保留原编号033疑似非真实食物图片的排除，主表99图，另给全部100图敏感性结果。粗粒度封闭词表、肉种和香草合并；不确定实体中立。无可见证据不等于已证明事实错误，不以参考caption自动构造视觉真值。标注未用于奖励或检查点选择，但历史上已观察过，结论属探索性。\n\n| 模型 | 食材精确率 % | 食材召回率 % | 无可见证据率 % | 遗漏率 % |\n|---|---:|---:|---:|---:|\n'
    for m,v in aggregate.items():text+=f"| {m} | {v['precision']*100:.2f} | {v['recall']*100:.2f} | {v['unsupported_rate']*100:.2f} | {v['omission_rate']*100:.2f} |\n"
    text+='\n逐图结果、5000次配对bootstrap及100图敏感性见entity_metrics.json和entity_per_image.jsonl。cases.jsonl按“无证据实体数+遗漏实体数”的减少值选择最大改善3例和最大退化3例，为看结果后选例，不能用6例代表总体发生率。\n\n## 典型案例\n\n'
    for case in selected:
        text+=f"### {case['image_id']}\n\n本协议下实体错误数减少{case['entity_error_count_reduction']}，选择规则：{case['selection']}。\n\n"
        for m,g in case['methods'].items():text+=f"- {m}：{g['caption']}；无可见证据实体：{', '.join(g['unsupported']) or '无'}；遗漏：{', '.join(g['omitted']) or '无'}。\n"
        text+='\n'
    text+='## 结论与局限\n\n'
    delta=test_ci['scst_minus_ce_original']['delta'];ed=ci['scst_minus_ce_original']['unsupported_rate']['delta']
    text+=f"本轮SCST相对原CE的测试CIDEr-D点估计{'上升' if delta>0 else '下降'}{abs(delta):.4f}，无可见证据率点估计{'上升' if ed>0 else '下降'}{abs(ed)*100:.2f}个百分点。数值奖励优化不自动保证视觉事实改善，需同时参考完整文本指标、重复率、实体指标和区间。\n\n本轮完成固定预算的最小完整实验，训练规模为2048图×3轮，不能写成全量训练或多种子稳定收益。无需因负结果继续在测试集调参。更大训练预算、多种子、独立真人盲标属于后续增强，不混入本轮结论。大权重与特征缓存不提交普通Git，检查点哈希和原始输入清单保留以追溯。\n\n方法依据：[Rennie等，CVPR 2017，Self-Critical Sequence Training](https://openaccess.thecvf.com/content_cvpr_2017/html/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.html)。\n"
    (out.parent/'REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps({'test':results,'entity':aggregate,'paired':test_ci,'complete':True},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
