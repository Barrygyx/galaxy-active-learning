"""Held-out Galaxy10 test evaluation for the frozen image_2000 campaign.
No training, model selection, or ensemble weight tuning.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

MEMBERS = ['resnet18_partial_aug', 'mobilenet_small_aug', 'efficientnet_b0_aug']
ALL_MODELS = ['resnet18_frozen', *MEMBERS]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--campaign', type=Path, help='Default: outputs/latest_image_2000.json')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    p.add_argument('--overwrite', action='store_true', help='Explicitly replace an earlier test report')
    args = p.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        p.error('batch-size and threads must be positive')
    root = args.project_dir.resolve()
    os.environ['TORCH_HOME'] = str(root / 'cache' / 'torch')
    engine = root / 'image_budget_runner'
    if not (engine / 'galaxy_image_demo' / 'models.py').is_file():
        raise FileNotFoundError(f'Missing bundled model code: {engine}')
    sys.path.insert(0, str(engine))
    import torch
    from galaxy_image_demo.data import load_prepared, cache_images
    from galaxy_image_demo.models import build_model
    from galaxy_image_demo.training import predict, metrics

    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu')

    if args.campaign:
        campaign = args.campaign if args.campaign.is_absolute() else root / args.campaign
    else:
        campaign = Path(read_json(root / 'outputs/latest_image_2000.json')['campaign_directory'])
        if not campaign.is_absolute():
            campaign = root / campaign
    campaign = campaign.resolve()
    registry = read_json(campaign / 'runs.json')['runs']
    if not registry:
        raise ValueError('No completed runs in campaign registry')
    campaign_protocol = read_json(campaign / 'campaign_protocol.json')
    if campaign_protocol.get('test_evaluated') is not False:
        raise ValueError('Campaign protocol does not designate test as held out')
    if campaign_protocol.get('ensemble_members') != MEMBERS:
        raise ValueError('Unexpected ensemble membership; do not silently change the final ensemble')
    if campaign_protocol.get('settings', {}).get('train_size') != 2000:
        raise ValueError('This evaluator expects the final 2000-label experiment')
    output = root / 'outputs' / 'final_test' / campaign.name
    if (output / 'test_results.json').exists() and not args.overwrite:
        raise FileExistsError(f'Test already reported at {output}. Avoid repeated test-driven selection; use --overwrite only if correcting an error.')

    config = read_json(Path(campaign_protocol['config']))
    data = load_prepared(root, config, smoke=False)
    test_positions = data.positions('test')
    if not len(test_positions):
        raise ValueError('Empty test split')
    test_ids = data.ids[test_positions]
    y_true = data.labels[test_positions]
    print(f'Campaign: {campaign}\nHeld-out test: {len(test_ids)} images; device={device}', flush=True)

    image_size = int(campaign_protocol['settings']['image_size'])
    images, positions, _, _ = cache_images(root, data, test_positions, image_size, torch)
    if not np.array_equal(positions, test_positions):
        raise ValueError('Cached test positions do not match original test split')
    indices = np.arange(len(test_positions), dtype=np.int64)
    all_metrics, class_rows, prediction_rows, cm_by_seed = [], [], [], {}
    checkpoint_hashes = {}
    per_seed_probabilities = {}
    for seed_str, record in sorted(registry.items(), key=lambda item: int(item[0])):
        seed = int(seed_str)
        exp = Path(record['experiment_directory'])
        if not exp.is_absolute():
            exp = root / exp
        protocol = read_json(exp / 'locked_protocol.json')
        if protocol.get('smoke') or protocol.get('test_evaluated') is not False:
            raise ValueError(f'Seed {seed}: not a held-out non-smoke experiment')
        if protocol['split_sha256'] != data.split_sha256 or protocol['settings']['seed'] != seed:
            raise ValueError(f'Seed {seed}: split or seed mismatch')
        if protocol['train_size'] != 2000 or protocol['image_size'] != image_size:
            raise ValueError(f'Seed {seed}: training budget or image size mismatch')
        probs = {}
        for name in ALL_MODELS:
            folder = exp / 'models' / name
            result = read_json(folder / 'result.json')
            ckpt_path = folder / 'best_model.pt'
            digest = sha256(ckpt_path)
            if digest != result['artifact_sha256']['best_model.pt']:
                raise ValueError(f'Checkpoint changed: {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            if checkpoint['model_name'] != name or checkpoint['identity'] != result['identity']:
                raise ValueError(f'Checkpoint identity mismatch: {ckpt_path}')
            model, _ = build_model(name, seed=seed, download=False)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.to(device)
            print(f'Seed {seed}: {name}', flush=True)
            probs[name] = predict(model, images, indices, args.batch_size, device, features=False)
            checkpoint_hashes[f'{seed}/{name}'] = digest
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        probs['equal_ensemble_3'] = np.mean(np.stack([probs[n] for n in MEMBERS], axis=0), axis=0)
        per_seed_probabilities[seed] = probs
        cm_by_seed[seed] = {}
        for name, prob in probs.items():
            if prob.shape != (len(y_true), 10) or not np.isfinite(prob).all() or not np.allclose(prob.sum(1), 1, atol=1e-5):
                raise ValueError(f'Invalid probabilities: {seed}/{name}')
            m = metrics(y_true, prob)
            row = {'seed': seed, 'model': name, 'n_test': len(y_true), 'accuracy': m['accuracy'], 'macro_f1': m['macro_f1'], 'cross_entropy': m['cross_entropy']}
            all_metrics.append(row)
            cm_by_seed[seed][name] = m['confusion_matrix']
            for c in m['per_class']:
                class_rows.append({'seed': seed, 'model': name, **c})
            pred = prob.argmax(axis=1)
            for i, source_id in enumerate(test_ids):
                prediction_rows.append({'seed': seed, 'model': name, 'source_index': int(source_id), 'true_label': int(y_true[i]),
                                        'pred_label': int(pred[i]), **{f'p_class_{j}': float(prob[i, j]) for j in range(10)}})
            print(f"  accuracy={m['accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'per_seed_metrics.csv', all_metrics)
    write_csv(output / 'per_class_metrics.csv', class_rows)
    write_csv(output / 'predictions.csv', prediction_rows)
    summary = []
    for name in [*ALL_MODELS, 'equal_ensemble_3']:
        rows = [r for r in all_metrics if r['model'] == name]
        s = {'model': name, 'n_seeds': len(rows)}
        for metric in ['accuracy', 'macro_f1', 'cross_entropy']:
            values = np.asarray([r[metric] for r in rows], dtype=float)
            s[f'{metric}_mean'] = float(values.mean())
            s[f'{metric}_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(s)
    write_csv(output / 'summary.csv', summary)
    for seed, matrices in cm_by_seed.items():
        for name, cm in matrices.items():
            np.savetxt(output / f'confusion_seed_{seed}_{name}.csv', np.asarray(cm, dtype=int), fmt='%d', delimiter=',')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for seed, matrices in cm_by_seed.items():
            cm = np.asarray(matrices['equal_ensemble_3'], dtype=int)
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(cm)
            fig.colorbar(im, ax=ax)
            ax.set(xlabel='Predicted class', ylabel='True class', title=f'Test Ensemble Confusion Matrix - Seed {seed}')
            ax.set_xticks(range(10)); ax.set_yticks(range(10))
            fig.tight_layout()
            fig.savefig(output / f'confusion_ensemble_seed_{seed}.png', dpi=170)
            plt.close(fig)
    except ImportError:
        print('matplotlib unavailable: saved confusion CSV only', flush=True)
    payload = {'campaign': str(campaign), 'split_sha256': data.split_sha256,
               'test_source_indices_sha256': hashlib.sha256(test_ids.tobytes()).hexdigest(),
               'n_test': len(y_true), 'seeds': sorted(per_seed_probabilities),
               'ensemble_members': MEMBERS, 'ensemble_rule': 'equal mean of three model probabilities within each seed',
               'checkpoints_sha256': checkpoint_hashes, 'per_seed': all_metrics, 'summary': summary}
    write_json(output / 'test_results.json', payload)
    lines = ['# Galaxy10 Held-Out Test Evaluation', '', f'Test images: {len(y_true)}; training label budget: 2000 per model.',
             'No test-set training, hyperparameter tuning, or ensemble weight fitting.', '',
             '| Model | Test Accuracy (mean ± SD) | Test Macro F1 (mean ± SD) |', '|---|---:|---:|']
    for s in summary:
        lines.append(f"| {s['model']} | {s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f} | {s['macro_f1_mean']:.4f} ± {s['macro_f1_std']:.4f} |")
    lines += ['', 'SD is across experiment seeds, not a test-sampling confidence interval.',
              'Each seed ensemble averages ResNet18 Partial, MobileNet Small and EfficientNet-B0 probabilities equally.',
              '', 'See per_seed_metrics.csv, per_class_metrics.csv, predictions.csv, and confusion matrices.']
    (output / 'test_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nDONE: {output / "test_report.md"}', flush=True)


if __name__ == '__main__':
    main()
