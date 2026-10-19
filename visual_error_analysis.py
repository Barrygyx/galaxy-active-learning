"""Read-only Galaxy10 visual error analysis using saved held-out predictions."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def montage(rows, h5, dest, title, n=20, seed=42):
    rows = rows.drop_duplicates('source_index').sort_values('source_index')
    if rows.empty:
        print('SKIP (no examples):', title)
        return
    rng = np.random.default_rng(seed)
    chosen = rows.iloc[np.sort(rng.choice(len(rows), min(n,len(rows)), replace=False))]
    cols = 5
    rr = int(np.ceil(len(chosen)/cols))
    fig, axes = plt.subplots(rr, cols, figsize=(cols*3.2,rr*3.35), squeeze=False)
    for ax in axes.flat: ax.axis('off')
    for ax, (_, r) in zip(axes.flat, chosen.iterrows()):
        idx=int(r.source_index)
        ax.imshow(h5['images'][idx]); ax.set_title(f'ID {idx} | true {int(r.true_label)} → pred {int(r.pred_label)}', fontsize=8)
        ax.axis('off')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0,0,1,.965));fig.savefig(dest,dpi=145);plt.close(fig)
    print('SAVED',dest,'examples=',len(chosen))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--predictions',type=Path,help='Path to predictions.csv; defaults to outputs/final_test/.../predictions.csv')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--examples',type=int,default=20)
    a=p.parse_args()
    root=a.project_dir.resolve()
    if a.predictions:
        predictions=a.predictions if a.predictions.is_absolute() else root/a.predictions
    else:
        candidates=sorted((root/'outputs/final_test').glob('*/predictions.csv'))
        if len(candidates)!=1:
            raise RuntimeError(f'Found {len(candidates)} prediction files. Specify --predictions explicitly.')
        predictions=candidates[0]
    out=predictions.parent/'visual_error_analysis'/f'seed_{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(predictions,encoding='utf-8-sig')
    d=df[df.seed==a.seed].copy()
    ensemble=d[d.model=='equal_ensemble_3'].copy()
    efficient=d[d.model=='efficientnet_b0_aug'].copy()
    if ensemble.empty or efficient.empty: raise ValueError('Missing ensemble or EfficientNet predictions for seed')
    if ensemble.source_index.duplicated().any() or efficient.source_index.duplicated().any(): raise ValueError('Duplicate source indices')
    compare=ensemble.merge(efficient[['source_index','true_label','pred_label']],on='source_index',suffixes=('','_efficient'),validate='one_to_one')
    if not (compare.true_label==compare.true_label_efficient).all():raise ValueError('Model labels do not agree')
    compare['ensemble_correct']=compare.true_label==compare.pred_label
    compare['efficient_correct']=compare.true_label==compare.pred_label_efficient
    compare['outcome']=np.select([compare.ensemble_correct & ~compare.efficient_correct,~compare.ensemble_correct & compare.efficient_correct,compare.ensemble_correct & compare.efficient_correct],['ensemble_rescue','ensemble_regression','both_correct'],default='both_wrong')
    compare.to_csv(out/'model_comparison.csv',index=False,encoding='utf-8-sig')
    confusion=pd.crosstab(ensemble.true_label,ensemble.pred_label).reindex(index=range(10),columns=range(10),fill_value=0)
    confusion.to_csv(out/'confusion_counts.csv',encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(confusion.values,cmap='Blues');fig.colorbar(im,ax=ax)
    ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted class',ylabel='True class',title=f'Ensemble confusion matrix, seed {a.seed}')
    for i in range(10):
        for j in range(10):
            v=int(confusion.iloc[i,j]);ax.text(j,i,str(v),ha='center',va='center',fontsize=8,color='white' if v>confusion.values.max()*.5 else 'black')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=170);plt.close(fig)
    config=json.loads((root/'al_config.json').read_text(encoding='utf-8-sig'))
    h5path=(root/config['data_file']).resolve()
    if not h5path.is_file():raise FileNotFoundError(h5path)
    with h5py.File(h5path,'r') as h5:
        if 'images' not in h5:raise ValueError('HDF5 images dataset missing')
        specs=[
          ('class0_correct',compare[(compare.true_label==0)&compare.ensemble_correct]),
          ('class0_as7',compare[(compare.true_label==0)&(compare.pred_label==7)]),
          ('class0_as3',compare[(compare.true_label==0)&(compare.pred_label==3)]),
          ('class4_correct',compare[(compare.true_label==4)&compare.ensemble_correct]),
          ('class4_wrong',compare[(compare.true_label==4)&~compare.ensemble_correct]),
          ('class6_as7',compare[(compare.true_label==6)&(compare.pred_label==7)]),
          ('class5_as7',compare[(compare.true_label==5)&(compare.pred_label==7)]),
          ('ensemble_rescue',compare[compare.outcome=='ensemble_rescue']),
          ('ensemble_regression',compare[compare.outcome=='ensemble_regression']),
        ]
        for name,rows in specs:
            montage(rows,h5,out/f'{name}.png',name.replace('_',' ').title()+f' | seed {a.seed}',a.examples,a.seed)
    print('\nOutcome counts:',compare.outcome.value_counts().to_dict())
    print('OUTPUT:',out)

if __name__=='__main__':main()
