"""Recompute dependent metrics, document correction scope, package a versioned handoff."""
from pathlib import Path
from collections import Counter
import json, hashlib, zipfile, shutil
import numpy as np
from metrics import ingredient_f1, NUM_KEYS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'final_results_corrected_20261119'
def rows(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    vocab=set(json.loads((ROOT/'ingredients.json').read_text()))
    train=rows(ROOT/'annotations/train.jsonl')
    std=np.std([[r['target'][k] for k in NUM_KEYS] for r in train],axis=0)
    health={};updated={}
    for split in ['val','test']:
        refs=rows(ROOT/f'annotations/{split}.jsonl')
        for tag in ['A','B']:
            pp=rows(OUT/f'model{tag}_{split}_preds.jsonl');assert [r['dish_id'] for r in pp]==[r['dish_id'] for r in refs]
            c=Counter()
            for r in pp:
                obj=r['parsed'];ns=[it['name'] for it in obj['items']]
                c['oov_samples']+=bool(set(ns)-vocab);c['duplicate_samples']+=len(ns)>len(set(ns))
                c['weight_diff_gt5g']+=abs(sum(it['weight_g'] for it in obj['items'])-obj['total_weight_g'])>5
                c['regex_fallback_samples']+=r.get('parse_code')==3
            health[f'{split}_{tag}']={'n':len(pp),**c}
        metrics=json.loads((OUT/f'{split}_metrics.json').read_text())
        for m in metrics.values():
            m['numeric']['normalized_mae_over_train_std']=float(np.mean([m['numeric'][k]['mae']/max(std[i],1e-8) for i,k in enumerate(NUM_KEYS)]))
        (OUT/f'{split}_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
        updated[split]=metrics
    # B-gen character decoder is unaffected. Correct macro for every preserved variant
    # from its own saved predictions, leaving other historical metrics unchanged.
    for source in sorted((ROOT/'final_results').glob('bgen_metrics*.json')):
        suffix=source.stem[len('bgen_metrics'):];data=json.loads(source.read_text())
        for split in ['val','test']:
            pp=rows(ROOT/f'final_results/modelBgen_{split}_preds{suffix}.jsonl')
            refs={r['dish_id']:r for r in rows(ROOT/f'annotations/{split}.jsonl')}
            m=ingredient_f1([[it['name'] for it in r['parsed']['items']] for r in pp],[[it['name'] for it in refs[r['dish_id']]['target']['items']] for r in pp])
            data[split]['metrics'].update(m)
        data['correction_note']='Macro denominator repaired; predictions and other metrics unchanged. JSON text metrics are custom implementations, not certified COCO scorer results.'
        (OUT/source.name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'health.json').write_text(json.dumps(health,indent=2),encoding='utf-8')
    old=json.loads((ROOT/'final_results/test_metrics.json').read_text())
    lines=['# Task2 2026-11-19 修正交付','',
        '本版更正旧提交中的解码、截断、Macro-F1与混合单位指标说明。原权重不变，A/B重新推理414条验证样本及506条测试样本；B-gen各已有版本按原预测重新计算食材F1。测试集属于历史已查看集，本次是软件勘误复算，不是新的盲测。','',
        '## 实际修复',
        '- A恢复字符串内多词名称和括号前空格，遇EOS停止；2338条训练目标JSON编码/解码回归全部通过。未按测试真值改名，也未将未知名字强行映射词表。',
        '- B默认先按置信度选最高16项，再按词表顺序输出；相同概率稳定排序。原阈值保持不变。旧行为仅显式传b_rank_overflow=False时保留。',
        '- Macro-F1对本划分中预测/真值名称并集求均值，包含零F1类别，报告类别数。micro-F1公式未变。',
        '- 后续训练按mean(MAE_j / train_std_j)选最佳模型，输出至checkpoints_lora_mt_normalized，禁止接续旧尺度选型断点。已有checkpoints_lora_mt权重仍按旧混合单位MAE选出，不能宣称已重新选型；目录仅剩best权重，无法追溯重新选择各轮。此次未重新训练。',
        '- 每字段单独报告MAE/MSE；混合单位mean_mae/mean_mse仅为兼容字段，不能解释成g/kcal。JSON文本指标为项目自定义实现，不等同于官方COCO评测器，固定JSON结构会提高匹配得分。','',
        '## 506条历史测试集复算','',
        '| 模型 | 旧micro-F1 | 新micro-F1 | 旧Macro-F1（错误口径） | 新Macro-F1 |','|---|---:|---:|---:|---:|']
    for key,m in updated['test'].items():
        o=old[key];lines.append(f"| {key} | {o['micro_f1']:.4f} | {m['micro_f1']:.4f} | {o['macro_f1']:.4f} | {m['macro_f1']:.4f} |")
    lines+=['','| 模型 | 重量MAE(g) | 热量MAE(kcal) | 蛋白质MAE(g) | 脂肪MAE(g) | 碳水MAE(g) |','|---|---:|---:|---:|---:|---:|']
    for key,m in updated['test'].items():lines.append('| '+key+' | '+' | '.join(f"{m['numeric'][k]['mae']:.3f}" for k in NUM_KEYS)+' |')
    lines+=['','## 仍然存在的模型能力问题','', '| 模型 | 词表外样本 | 重复食材样本 | 分项总和偏差>5g | 正则回退样本 |','|---|---:|---:|---:|---:|']
    for tag in ['A','B']:
        h=health['test_'+tag];lines.append(f"| {tag} | {h['oov_samples']} | {h['duplicate_samples']} | {h['weight_diff_gt5g']} | {h['regex_fallback_samples']} |")
    lines+=['','解码修复不等于模型预测全部正确。重复食材、重量不一致、回退解析按原始输出如实披露；没有为降低统计数字强行去重、补答案或修改预测总量。B为空食材但预测总量非零的样本仍可能重量不一致。',
        '', '## 复现与交付',
        '`python -m unittest test_task2_corrections -v`；`python eval_models.py`输出final_results_corrected_20261119。三项回归检查通过；本轮权重SHA256见correction_manifest.json。',
        '更正版ZIP保留模型和运行所需文件；旧报告/结果移入historical_reports和historical_results，标为历史，报告数字以本文件及final_results中的更正结果为准。阶段历史材料无需假装是当时已修正的版本。',
        '本次未改附加挑战已有校准阈值。涉及源码哈希校验的旧冻结实验可能因本次代码修正拒绝运行，应使用已备份旧代码复现历史结果或登记新版本，不能直接篡改旧签名。']
    report=OUT/'修正说明与结果.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    code=['model_caption.py','metrics.py','predict.py','eval_models.py','train_lora_mt.py','test_task2_corrections.py','finalize_task2_correction.py']
    manifest={'version':'20261119-correction','weights_changed':False,'retrained':False,'regression_tests_passed':3,
              'code_sha256':{n:sha(ROOT/n) for n in code},'checkpoint_sha256':{n:sha(ROOT/n) for n in ['checkpoints_caption/best_model.pt','checkpoints_lora_mt/best_model.pt']}}
    (OUT/'correction_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    source=ROOT/'FoodLens_Task2_1109_最终交付_去重版.zip';dest=ROOT/'FoodLens_Task2_1109_最终交付_20261119更正版.zip'
    with zipfile.ZipFile(source) as zin,zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as zout:
        prefix=next(n for n in zin.namelist() if n.endswith('/model_caption.py'))[:-len('model_caption.py')]
        for info in zin.infolist():
            if info.is_dir():continue
            n=info.filename
            if n.startswith(prefix):
                rel=n[len(prefix):]
                if rel in code:continue
                if rel.startswith('final_results/'):n=prefix+'historical_results/'+rel[len('final_results/'):]
                elif rel.endswith('.docx'):n=prefix+'historical_reports/'+rel
            with zin.open(info) as src,zout.open(n,'w',force_zip64=True) as dst:shutil.copyfileobj(src,dst,1024*1024)
        for n in code:zout.write(ROOT/n,prefix+n)
        for p in OUT.iterdir():
            if p.is_file():zout.write(p,prefix+'final_results/'+p.name)
        zout.write(report,'请先阅读_20261119修正说明.md')
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    assert dest.stat().st_size<1_000_000_000
    print('PACKAGE',dest.name,dest.stat().st_size,flush=True)
    print(json.dumps({'health':health,'test':{k:{x:v[x] for x in ['micro_f1','macro_f1']} for k,v in updated['test'].items()}},indent=2))

if __name__=='__main__':main()
