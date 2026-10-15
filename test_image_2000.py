"""Focused checks for cross-seed aggregation, provenance and reporting failures."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from run_image_2000 import MODELS,MEMBERS,write_json,digest,parser
from summarize_image_2000 import check_metrics,collect,summarize,main


def measured(perfect):
    cm=[[0]*10 for _ in range(10)]
    for i in range(10):
        cm[i][i if perfect else 0]=1
    per=[]
    for i in range(10):
        tp=cm[i][i];pred=sum(row[i] for row in cm)
        per.append({'class_id':i,'precision':tp/pred if pred else 0.,'recall':float(tp),
                    'f1':2*tp/(1+pred),'support':1})
    return {'accuracy':sum(cm[i][i] for i in range(10))/10,'macro_f1':sum(p['f1'] for p in per)/10,
            'per_class':per,'confusion_matrix':cm,'cross_entropy':.1 if perfect else 2.3}


def fixture(root,seeds=(42,43)):
    settings={'train_size':10,'monitor_size':10,'image_size':64,'epochs':2,'patience':1,
              'min_epochs':1,'warmup_epochs':0,'batch_size':4}
    write_json(root/'campaign_protocol.json',{'settings':settings})
    registry={'runs':{}}
    for seed in seeds:
        exp=root/f'fit_{seed}';exp.mkdir()
        proto={'smoke':False,'test_evaluated':False,'settings':{'seed':seed,**settings},
               'train_size':10,'image_size':64,'split_sha256':'same-original-split'}
        write_json(exp/'locked_protocol.json',proto)
        with (exp/'subsets.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['role','source_index','label']);w.writeheader()
            for role in ['train','monitor','validation']:
                for i in range(10):
                    w.writerow({'role':role,'source_index':seed*10+i if role=='train' else 1000+i,'label':i})
        results={}
        for model in MODELS:
            folder=exp/'models'/model;folder.mkdir(parents=True)
            h=[{'epoch':i,'train_monitor_macro_f1':.3,'monitor_macro_f1':.2,
                'train_monitor_ce':1.2,'monitor_ce':1.5,'monitor_class4_f1':0.,'monitor_class5_f1':.2} for i in [1,2]]
            write_json(folder/'history.json',h)
            for name in ['best_model.pt','predictions.npz']:
                (folder/name).write_bytes(b'fixture artifact')
            result={'identity':{'model_name':model,'pilot_protocol_sha256':hashlib.sha256(json.dumps(proto,sort_keys=True).encode()).hexdigest()},
                    'train':measured(True),'validation':measured(model!='resnet18_frozen'),
                    'best_epoch':2,'epochs_run':2,'stop_reason':'max_epochs','training_seconds':1.,
                    'artifact_sha256':{n:digest(folder/n) for n in ['best_model.pt','predictions.npz','history.json']}}
            write_json(folder/'result.json',result);results[model]=result
        ensemble=exp/'ensemble';ensemble.mkdir()
        write_json(ensemble/'ensemble_result.json',{'members':MEMBERS,'weights':[1/3]*3,'training_labels':10,
            'member_prediction_sha256':{n:results[n]['artifact_sha256']['predictions.npz'] for n in MEMBERS},
            'test_evaluated':False,'train':measured(True),'validation':measured(True)})
        registry['runs'][str(seed)]={'seed':seed,'experiment_directory':str(exp),'ensemble_directory':str(ensemble)}
    write_json(root/'runs.json',registry)


class BudgetTests(unittest.TestCase):
    def test_same_budget_protocol_defaults(self):
        args=parser().parse_args([])
        self.assertEqual((args.train_size,args.epochs,args.monitor_size,args.patience,args.min_epochs,args.warmup_epochs),
                         (2000,30,800,6,8,2))

    def test_macro_metrics_include_failed_classes(self):
        m=measured(False);check_metrics(m)
        self.assertAlmostEqual(m['accuracy'],.1)
        self.assertAlmostEqual(m['macro_f1'],(2/11)/10)
        m['per_class'][4]['f1']=.5
        with self.assertRaisesRegex(ValueError,'metric/confusion'):
            check_metrics(m)

    def test_cross_seed_summary_and_ensemble_supports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            rows,classes,counts,_,_,_=collect(root)
            self.assertEqual(len(rows),20)
            self.assertEqual(len(classes),200)
            self.assertEqual(len(counts),60)
            summary=summarize(rows)
            self.assertTrue(all(r['n_seeds']==2 for r in summary))
            self.assertEqual(summary[-1]['macro_f1_mean'],1.)
            altered=copy.deepcopy(rows)
            for row in altered:
                if row['model']=='equal_ensemble_3' and row['split']=='validation' and row['seed']==43:
                    row['macro_f1']=.6
            r=summarize(altered)[-1]
            self.assertAlmostEqual(r['macro_f1_mean'],.8)
            self.assertAlmostEqual(r['macro_f1_std'],(.08)**.5)

    def test_modified_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root,(42,))
            (root/'fit_42/models/resnet18_frozen/predictions.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed artifact'):
                collect(root)

    def test_different_full_validation_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            p=root/'fit_43/subsets.csv';s=p.read_text();p.write_text(s.replace('1000','2000'))
            with self.assertRaisesRegex(ValueError,'validation images differ'):
                collect(root)

    def test_report_exports_expected_rows_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root,(42,))
            main(['--campaign',str(root)])
            with (root/'per_class_metrics.csv').open(encoding='utf-8-sig') as f:
                self.assertEqual(len(list(csv.DictReader(f))),100)
            self.assertEqual(len(list((root/'figures').glob('*.png'))),5)
            self.assertTrue((root/'SUMMARY_COMPLETE.json').exists())
            self.assertIn('训练标签=10', (root/'image_2000_report.md').read_text())


if __name__=='__main__':
    unittest.main()
