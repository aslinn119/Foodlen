"""Verify final protocol outputs and create a self-contained, <1GB submission."""
from pathlib import Path
import json,zipfile,shutil,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'final_results_v3_20261119'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    freeze=json.loads((OUT/'freeze.json').read_text())
    for name,h in freeze['files'].items():assert sha(ROOT/name)==h,name
    history=json.loads((ROOT/'checkpoints_lora_mt_normalized/history.json').read_text())
    assert len(history['train_loss'])==25
    ck=torch.load(ROOT/'checkpoints_lora_mt_normalized/best_model.pt',map_location='cpu',weights_only=False)
    assert ck['selection_metric']=='mean_mae_over_train_std'
    assert np.isclose(ck['selection_score'],min(history['val_normalized_mae']))
    epoch=int(np.argmin(history['val_normalized_mae']))+1
    totals={}
    for split in ['val','test']:
        for model in ['a','b']:
            rr=[json.loads(l) for l in (OUT/f'{model}_{split}_predictions.jsonl').read_text().splitlines()]
            assert len(rr)==(414 if split=='val' else 506)
            refs=[json.loads(l) for l in (ROOT/f'annotations/{split}.jsonl').read_text().splitlines()]
            assert [r['dish_id'] for r in rr]==[r['dish_id'] for r in refs]
            for r in rr:
                p=r['parsed'];assert r['parse_code']==1
                assert len(p['items'])==len({i['name'] for i in p['items']})>0
                assert abs(sum(i['weight_g'] for i in p['items'])-p['total_weight_g'])<1e-6
                if model=='a':assert json.loads(r['text'])==r['raw_parsed']
                assert all(np.isfinite(p[k]) and p[k]>=0 for k in ['total_weight_g','total_calories_kcal','protein_g','fat_g','carbohydrate_g'])
            totals[model+'_'+split]=len(rr)
    text=['# FoodLens Task2 最终修复与训练交付（2026-11-19）','',
          '本版替代20261119更正版作为当前运行入口。已完成25轮新B训练、A约束解码、精确重量一致性处理、Macro-F1修正和参考文本指标评测。历史测试集已经被查看，以下是回溯修正评估，不能称新盲测。','',
          '## 新训练与选型',f'B完整训练25轮，按验证集mean(MAE_j/train_std_j)选择第{epoch}轮，最低无量纲得分{ck["selection_score"]:.6f}。训练标准差只由train计算。保留旧B权重以复现B-gen和历史实验，新权重位于checkpoints_lora_mt_normalized。',
          '阈值沿用所选轮次在验证集确定的值，超16项按概率保留最高项；无项超过阈值时选择概率最高的一项。该规则可能增加假阳性，必须结合精确率和召回率解释，不等于知道真实食材。','',
          '## A生成与结构一致性',
          'A仍使用图像编码器和GRU的预测logits，自回归生成token；合法token由固定JSON语法和训练食材词表限制。生成阶段禁止重复名字、禁止非法数值和非法JSON。不是根据测试标签查找答案。',
          '分项重量按模型预测总重投影到0.1g整数单位，最大余数法保证总和一致；原始生成文本和投影前结构均保存。重量一致性来自输出约束，不能当作重量估计准确率提升。A没有重新训练。','',
          '## 参考指标实现',
          '使用[pycocoevalcap 1.2](https://github.com/salaniz/pycocoevalcap)的BLEU、ROUGE-L、CIDEr评分代码；CIDEr使用裁剪与Gaussian长度惩罚。为了适配JSON，显式传入统一JSON词法tokens，而非Stanford PTB处理，属于“参考评分算法+明确的JSON分词协议”，不能写作完整COCO自然语言榜单口径。',
          'Macro-F1以当前划分预测/真值名称并集为类别范围，零F1类别计入。micro-F1独立计算。各数值分别报告单位；兼容输出中的混合单位mean_mae不是g或kcal。','',
          '## 历史测试集结果','',
          '| 模型 | BLEU-4 | ROUGE-L | CIDEr-D | micro-F1 | Macro-F1 | 精确率 | 召回率 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in ['a','b','bgen']:
        m=json.loads((OUT/f'{model}_test_metrics.json').read_text())
        text.append('| '+model+' | '+' | '.join(f'{m[k]:.4f}' for k in ['bleu_4','rouge_l','cider_d','micro_f1','macro_f1','micro_p','micro_r'])+' |')
    text+=['','| 模型 | 重量MAE(g) | 热量MAE(kcal) | 蛋白MAE(g) | 脂肪MAE(g) | 碳水MAE(g) |','|---|---:|---:|---:|---:|---:|']
    for model in ['a','b','bgen']:
        m=json.loads((OUT/f'{model}_test_metrics.json').read_text())
        text.append('| '+model+' | '+' | '.join(f'{m["numeric"][k]["mae"]:.3f}' for k in ['total_weight_g','total_calories_kcal','protein_g','fat_g','carbohydrate_g'])+' |')
    text+=['','## 验收范围',
           'A/B各414条验证、506条测试输出：严格解析、食材唯一、重量相等检查全部通过。A原始生成JSON也全部严格解析。结构通过不代表图中食材全识别正确或五项数值都准确；以真实误差表为准。',
           'B-gen保留为旧生成模型对照，未伪装为新B权重版本，原预测不变，文本和食材指标按统一新实现重算。其原有输出缺陷必须继续保留在对照结果中。','',
           '## 使用',
           '当前A/B入口：`from predict import predict_nutrition`，或`from predict_final import predict_nutrition`。默认b加载新归一化选型权重。旧A/B通过predict.py的legacy=True显式调用。bgen始终保留原权重。',
           '运行 `python -m unittest test_task2_corrections -v`。评测 `python eval_task2_final.py --split val --model a`（b/bgen同理）。test评测要求freeze.json存在。原始评测脚本eval_models.py只用于旧权重对照，不是本版主评测入口。',
           'requirements_final.txt记录测试环境依赖，CUDA版torch需从PyTorch对应cu124源安装。完整源数据需按原数据授权单独准备；包保留已有demo图供接口调用。',
           '旧报告/结果仅留作历史，最新数字以本文件和final_results_v3_20261119为准。新增数据预留校准/测试集本轮没有使用；附加挑战旧校准阈值不适用于新B，需另行登记和校准。']
    report=OUT/'最终修复报告.md';report.write_text('\n'.join(text)+'\n',encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps({'passed':True,'prediction_counts':totals,'epochs':25,'selected_epoch':epoch,'selection_score':ck['selection_score'],'training_selection_matches_history':True,'freeze_hashes_match':True},indent=2))
    files=['predict.py','predict_final.py','constrained_caption.py','eval_task2_final.py','eval_models.py','train_lora_mt.py','metrics.py','model_caption.py','test_task2_corrections.py','requirements_final.txt','build_task2_v3.py','finish_task2_v3.py']
    src=ROOT/'FoodLens_Task2_1109_最终交付_20261119更正版.zip';dst=ROOT/'FoodLens_Task2_1109_最终交付_v3训练完成.zip'
    with zipfile.ZipFile(src) as zin,zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as zout:
        prefix=next(n for n in zin.namelist() if n.endswith('/model_caption.py'))[:-len('model_caption.py')]
        for item in zin.infolist():
            if item.is_dir():continue
            n=item.filename
            if n.startswith(prefix):
                rel=n[len(prefix):]
                if rel in files:continue
                if rel.startswith('final_results/'):n=prefix+'historical_v2_results/'+rel[len('final_results/'):]
            if n=='请先阅读_20261119修正说明.md':n='历史_第一轮修正说明.md'
            with zin.open(item) as a,zout.open(n,'w',force_zip64=True) as b:shutil.copyfileobj(a,b,1024*1024)
        for n in files:zout.write(ROOT/n,prefix+n)
        for p in OUT.iterdir():
            if p.is_file():zout.write(p,prefix+'final_results_v3_20261119/'+p.name)
        for n in ['best_model.pt','history.json']:zout.write(ROOT/'checkpoints_lora_mt_normalized'/n,prefix+'checkpoints_lora_mt_normalized/'+n)
        zout.write(report,'请先阅读_最终修复报告.md')
    with zipfile.ZipFile(dst) as z:assert z.testzip() is None;assert len(z.namelist())==len(set(z.namelist()))
    print('PACKAGE',dst.name,dst.stat().st_size,flush=True)
    assert dst.stat().st_size<1_000_000_000,'Need date-separated supplementary model package to stay below WeChat size limit'
if __name__=='__main__':main()
