"""Summarize completed seed runs. Validation only; no model or threshold tuning."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics
from run_image_2000 import MODELS, MEMBERS, read_json, write_json, digest

CLASS_NAMES = ['Disturbed','Merging','Round smooth','In-between round smooth','Cigar shaped smooth',
               'Barred spiral','Unbarred tight spiral','Unbarred loose spiral','Edge-on without bulge','Edge-on with bulge']
METHODS = [*MODELS,'equal_ensemble_3']
SHORT = ['Frozen ResNet18','Partial ResNet18','EfficientNet-B0','MobileNetV3 Small','Equal ensemble']


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('No rows to report.')
    tmp = Path(path).with_name(Path(path).name+'.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def check_metrics(m):
    """Verify reported accuracy/F1 and all ten supports against confusion counts."""
    cm = m['confusion_matrix']
    if len(cm)!=10 or any(len(row)!=10 for row in cm):
        raise ValueError('Expected ten-class confusion matrix.')
    if any(not isinstance(v,int) or v<0 for row in cm for v in row):
        raise ValueError('Invalid confusion counts.')
    per = sorted(m['per_class'],key=lambda x:x['class_id'])
    if [p['class_id'] for p in per] != list(range(10)):
        raise ValueError('Missing/duplicate classes.')
    f1s = []
    for i,p in enumerate(per):
        support, predicted, tp = sum(cm[i]),sum(cm[j][i] for j in range(10)),cm[i][i]
        precision = tp/predicted if predicted else 0.
        recall = tp/support if support else 0.
        f1 = 2*tp/(support+predicted) if support+predicted else 0.
        if p['support'] != support or any(abs(a-b)>1e-9 for a,b in [(p['precision'],precision),(p['recall'],recall),(p['f1'],f1)]):
            raise ValueError('Class metric/confusion mismatch.')
        f1s.append(f1)
    total = sum(map(sum,cm))
    if not total or abs(statistics.mean(f1s)-m['macro_f1'])>1e-9 or abs(sum(cm[i][i] for i in range(10))/total-m['accuracy'])>1e-9:
        raise ValueError('Overall metric/confusion mismatch.')
    return per


def collect(campaign):
    protocol = read_json(campaign/'campaign_protocol.json')
    registry = read_json(campaign/'runs.json')['runs']
    if not registry:
        raise ValueError('No completed seeds.')
    rows, class_rows, counts, histories, confusions, pairs = [], [], [], {}, {}, []
    split_sha = None
    validation_ids = None
    for key,entry in sorted(registry.items(),key=lambda x:int(x[0])):
        seed = int(key)
        if entry['seed'] != seed:
            raise ValueError('Seed registry mismatch.')
        folder = Path(entry['experiment_directory'])
        actual = read_json(folder/'locked_protocol.json')
        if actual['smoke'] or actual['test_evaluated'] or actual['settings']['seed']!=seed:
            raise ValueError('Expected a completed validation-only non-smoke run.')
        if actual['train_size']!=protocol['settings']['train_size'] or actual['image_size']!=protocol['settings']['image_size']:
            raise ValueError('Budget/resolution mismatch.')
        for name in ['epochs','patience','min_epochs','warmup_epochs','batch_size']:
            if actual['settings'][name]!=protocol['settings'][name]:
                raise ValueError(f'Training settings mismatch: {name}')
        if split_sha is not None and split_sha!=actual['split_sha256']:
            raise ValueError('Different original split between seeds.')
        split_sha = actual['split_sha256']
        with (folder/'subsets.csv').open(encoding='utf-8-sig',newline='') as f:
            subsets = list(csv.DictReader(f))
        id_sets = {s:{int(r['source_index']) for r in subsets if r['role']==s} for s in ['train','monitor','validation']}
        if id_sets['train'] & id_sets['validation'] or not id_sets['monitor']<=id_sets['validation']:
            raise ValueError('Train/validation leakage in recorded subsets.')
        if validation_ids is not None and validation_ids!=id_sets['validation']:
            raise ValueError('Full validation images differ between seeds.')
        validation_ids = id_sets['validation']
        for split in ['train','monitor','validation']:
            c = [sum(r['role']==split and int(r['label'])==i for r in subsets) for i in range(10)]
            if len(id_sets[split])!=sum(c):
                raise ValueError('Duplicate subset IDs.')
            counts.extend({'seed':seed,'split':split,'class_id':i,'class_name':CLASS_NAMES[i],'support':c[i]} for i in range(10))
        results = {}
        import hashlib,json
        protocol_digest = hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()
        for model in MODELS:
            mdir = folder/'models'/model
            r = read_json(mdir/'result.json')
            if r['identity']['model_name']!=model or r['identity']['pilot_protocol_sha256']!=protocol_digest:
                raise ValueError('Model identity mismatch.')
            for name,expected in r['artifact_sha256'].items():
                if digest(mdir/name)!=expected:
                    raise ValueError(f'Changed artifact: {mdir/name}')
            results[model] = r
            histories[(seed,model)] = read_json(mdir/'history.json')
        ensemble = read_json(Path(entry['ensemble_directory'])/'ensemble_result.json')
        if ensemble['members']!=MEMBERS or ensemble['test_evaluated'] or ensemble['training_labels']!=actual['train_size']:
            raise ValueError('Ensemble protocol mismatch.')
        if len(ensemble['weights'])!=3 or any(abs(w-1/3)>1e-12 for w in ensemble['weights']):
            raise ValueError('Expected fixed equal ensemble weights.')
        for member in MEMBERS:
            if ensemble['member_prediction_sha256'][member]!=results[member]['artifact_sha256']['predictions.npz']:
                raise ValueError('Ensemble predictions do not match completed models.')
        results['equal_ensemble_3'] = ensemble
        for model,r in results.items():
            for split in ['train','validation']:
                m = r[split]
                per = check_metrics(m)
                c = [x for x in counts if x['seed']==seed and x['split']==split]
                if [p['support'] for p in per]!=[p['support'] for p in c]:
                    raise ValueError('Evaluation supports differ from recorded subsets.')
                rows.append({'seed':seed,'model':model,'split':split,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],
                    'class4_f1':per[4]['f1'],'class4_recall':per[4]['recall'],'class5_f1':per[5]['f1'],
                    'class5_recall':per[5]['recall'],'cross_entropy':m['cross_entropy'],
                    'best_epoch':r.get('best_epoch','member checkpoints'),'epochs_run':r.get('epochs_run',''),
                    'stop_reason':r.get('stop_reason','ensemble; no new training'),
                    'training_seconds':r.get('training_seconds',sum(results[n]['training_seconds'] for n in MEMBERS)),
                    'train_val_macro_gap':r['train']['macro_f1']-r['validation']['macro_f1']})
                class_rows.extend({'seed':seed,'model':model,'split':split,'class_name':CLASS_NAMES[p['class_id']],**p} for p in per)
                if split=='validation':
                    confusions[(seed,model)] = m['confusion_matrix']
                    for i in [0,4,5,6]:
                        support = per[i]['support']
                        alternatives = sorted([j for j in range(10) if j!=i],key=lambda j:-m['confusion_matrix'][i][j])[:3]
                        for rank,j in enumerate(alternatives,1):
                            pairs.append({'seed':seed,'model':model,'true_class':i,'predicted_class':j,'rank':rank,
                                'count':m['confusion_matrix'][i][j], 'fraction_of_true_class':m['confusion_matrix'][i][j]/support if support else 0.})
    return rows,class_rows,counts,histories,confusions,pairs


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['split']=='validation':
            groups[row['model']].append(row)
    summary = []
    for model in METHODS:
        values = groups[model]
        item = {'model':model,'n_seeds':len(values)}
        for metric in ['accuracy','macro_f1','class4_f1','class4_recall','class5_f1','class5_recall','train_val_macro_gap']:
            v = [r[metric] for r in values]
            item[metric+'_mean'] = statistics.mean(v)
            item[metric+'_std'] = statistics.stdev(v) if len(v)>1 else 0.
        summary.append(item)
    return summary


def figures(campaign,rows,class_rows,counts,histories,confusions,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out = campaign/'figures'
    out.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(out/(name+'.png'),dpi=150)
        fig.savefig(out/(name+'.pdf'))
        plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(13,5))
    for ax,metric in zip(axes,['accuracy','macro_f1']):
        means = [s[metric+'_mean'] for s in summary]
        errs = [s[metric+'_std'] for s in summary]
        ax.bar(range(5),means,yerr=errs,capsize=3,color=['C0','C1','C2','C3','C4'])
        ax.set_xticks(range(5),SHORT,rotation=25,ha='right')
        ax.set_ylim(0,1)
        ax.set_ylabel('Full validation '+metric)
    budget=read_json(campaign/'campaign_protocol.json')['settings']['train_size']
    fig.suptitle(f"Same {budget} labels per model | {summary[0]['n_seeds']} seed(s) | bars: mean; error: sample SD")
    save(fig,'model_comparison')
    fig,axes = plt.subplots(1,2,figsize=(13,5))
    for ax,metric in zip(axes,['class4_f1','class5_f1']):
        ax.bar(range(5),[s[metric+'_mean'] for s in summary],yerr=[s[metric+'_std'] for s in summary],capsize=3)
        ax.set_xticks(range(5),SHORT,rotation=25,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Validation '+metric)
    save(fig,'rare_and_barred_f1')
    seeds = sorted({r['seed'] for r in rows})
    for seed in seeds:
        fig,ax = plt.subplots(figsize=(11,5))
        x = np.arange(10)
        for i,model in enumerate(METHODS):
            v = sorted([r for r in class_rows if r['seed']==seed and r['model']==model and r['split']=='validation'],key=lambda r:r['class_id'])
            ax.plot(x,[r['f1'] for r in v],marker='o',label=SHORT[i])
        ax.set_xticks(x,[f'{i}: {n}' for i,n in enumerate(CLASS_NAMES)],rotation=35,ha='right')
        ax.set_ylim(0,1);ax.set_ylabel('Full validation class F1');ax.set_title(f'Seed {seed} | ten classes');ax.legend(fontsize=8)
        save(fig,f'per_class_f1_seed_{seed}')
        fig,axes = plt.subplots(4,3,figsize=(15,13),squeeze=False)
        for i,model in enumerate(MODELS):
            h = histories[(seed,model)];e = [r['epoch'] for r in h]
            for key,label in [('train_monitor_macro_f1','Train monitor'),('monitor_macro_f1','Validation monitor')]:
                axes[i,0].plot(e,[r[key] for r in h],label=label)
            for key,label in [('train_monitor_ce','Train monitor'),('monitor_ce','Validation monitor')]:
                axes[i,1].plot(e,[r[key] for r in h],label=label)
            for key,label in [('monitor_class4_f1','Class 4'),('monitor_class5_f1','Class 5')]:
                axes[i,2].plot(e,[r[key] for r in h],label=label)
            selected = next(r['best_epoch'] for r in rows if r['seed']==seed and r['model']==model)
            for j,metric in enumerate(['Macro-F1','Cross entropy','Validation class F1']):
                ax=axes[i,j];ax.axvline(selected,color='gray',linestyle='--');ax.set_title(SHORT[i]);ax.set_xlabel('Epoch');ax.set_ylabel(metric);ax.legend(fontsize=8)
        save(fig,f'training_curves_seed_{seed}')
        fig,axes = plt.subplots(2,3,figsize=(15,9))
        for i,model in enumerate(METHODS):
            ax=axes.flat[i];cm=np.asarray(confusions[(seed,model)]);normalized=cm/np.maximum(cm.sum(1,keepdims=True),1)
            ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
            ax.set_xticks(range(10));ax.set_yticks(range(10));ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(SHORT[i])
            for a in range(10):
                for b in range(10):
                    if normalized[a,b]>=.10:
                        ax.text(b,a,f'{normalized[a,b]*100:.0f}',ha='center',va='center',fontsize=7,color='white' if normalized[a,b]>.5 else 'black')
        axes.flat[-1].axis('off')
        fig.suptitle(f'Seed {seed}: full validation confusion, row percentages')
        save(fig,f'confusions_seed_{seed}')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True,type=Path)
    args=p.parse_args(argv);campaign=args.campaign.resolve()
    rows,classes,counts,histories,confusions,pairs=collect(campaign)
    summary=summarize(rows)
    write_csv(campaign/'metrics_by_seed.csv',rows)
    write_csv(campaign/'per_class_metrics.csv',classes)
    write_csv(campaign/'class_counts.csv',counts)
    write_csv(campaign/'metrics_mean_std.csv',summary)
    write_csv(campaign/'top_confusions.csv',pairs)
    baseline={r['seed']:r for r in rows if r['model']=='resnet18_frozen' and r['split']=='validation'}
    paired=[{'seed':r['seed'],'model':r['model'],**{m+'_difference':r[m]-baseline[r['seed']][m] for m in ['accuracy','macro_f1','class4_f1','class4_recall','class5_f1']}} for r in rows if r['split']=='validation' and r['model']!='resnet18_frozen']
    write_csv(campaign/'paired_differences_vs_frozen.csv',paired)
    figures(campaign,rows,classes,counts,histories,confusions,summary)
    protocol=read_json(campaign/'campaign_protocol.json');s=protocol['settings'];n=summary[0]['n_seeds']
    lines=['# Galaxy10 同预算图像模型对照','',f"训练标签={s['train_size']}；分辨率={s['image_size']}；最大 epochs={s['epochs']}；patience={s['patience']}；已完成 seeds={n}。",'',
        '这是模型开发实验。所有单模型在每个 seed 内共用相同训练图、验证 monitor 和训练规则；原 test 未评估。',
        '训练图仍从原 train_pool 均匀随机抽取，未使用隐藏类别挑选初始标签。跨 seed 会改变训练子集、初始化及 monitor。',
        '集成对三个预先指定的微调模型概率等权平均，标签预算不变，但计算成本高于单模型。','',
        '| 方法 | Accuracy mean | Macro-F1 mean ± SD | Class 4 F1 mean | Class 4 recall mean | Class 5 F1 mean | Train−val F1 gap mean |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['model']} | {r['accuracy_mean']:.4f} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | {r['class4_f1_mean']:.4f} | {r['class4_recall_mean']:.4f} | {r['class5_f1_mean']:.4f} | {r['train_val_macro_gap_mean']:.4f} |")
    lines+=['','单 seed 的 SD 显示 0 只是没有重复测量，不能解释为结果稳定。多 seed 的 SD 是样本标准差，不是置信区间或显著性检验。','',
        '## 每个模型的停止情况','', '| Seed | 方法 | 最佳 epoch | 已训练 epochs | 停止原因 |','|---:|---|---:|---:|---|']
    for r in rows:
        if r['split']=='validation' and r['model'] in MODELS:
            lines.append(f"| {r['seed']} | {r['model']} | {r['best_epoch']} | {r['epochs_run']} | {r['stop_reason']} |")
    lines+=['','## 稀有类支持数','', '| Seed | Split | Class 4 图数 | Class 5 图数 |','|---:|---|---:|---:|']
    for seed in sorted(baseline):
        for split in ['train','monitor','validation']:
            c={r['class_id']:r['support'] for r in counts if r['seed']==seed and r['split']==split}
            lines.append(f"| {seed} | {split} | {c[4]} | {c[5]} |")
    lines+=['','## 解释边界','',
        '- 同预算对照主要看 paired_differences_vs_frozen.csv；class 4 和总体 F1 必须一起检查。',
        '- 相比旧 512-label demo，本轮同时增加标签、训练上限和验证 monitor，不能把变化单独归因于其中一个因素。',
        '- 早停 monitor 包含在完整 validation 中，完整 validation 也用于比较方法；没有独立确认成绩。',
        '- 到达 max_epochs 不代表模型已经收敛；早停也不能保证消除过拟合。观察训练曲线、最佳 epoch 和逐类结果。',
        '- 训练 F1 来自恢复后的完整训练子集；曲线中的 train monitor 通常只有 256 张，因此终点数值可不同。',
        '- 该实验尚未执行主动学习；不能据此宣称节省标签、达到论文 accuracy 或证明某种选样策略有效。',
        '- 该流程读取原清洗审计记录中的标签和哈希以验证文件，但模型拟合、归一化和训练权重只使用当前训练子集。','',
        '## 回传结果','', '- image_2000_report.md','- metrics_by_seed.csv','- per_class_metrics.csv','- class_counts.csv','- figures/training_curves_seed_42.png','- figures/confusions_seed_42.png','']
    (campaign/'image_2000_report.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(campaign/'SUMMARY_COMPLETE.json',{'test_evaluated':False,'n_seeds':n,'seeds':sorted(baseline),
        'metrics_reconciled_to_confusions':True,'model_artifact_hashes_verified':True,'train_validation_disjoint':True})
    print(f'SUMMARY: {campaign/"image_2000_report.md"}',flush=True)


if __name__=='__main__':
    main()
