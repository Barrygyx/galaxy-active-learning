"""Run a shared-budget image comparison using the isolated, bundled CNN runner."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

MODELS = ['resnet18_frozen', 'resnet18_partial_aug', 'efficientnet_b0_aug', 'mobilenet_small_aug']
MEMBERS = ['resnet18_partial_aug', 'mobilenet_small_aug', 'efficientnet_b0_aug']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def stream_process(command, log_path, cwd):
    """Print progress live, retain an append-only log, propagate child failure."""
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    with Path(log_path).open('a', encoding='utf-8') as log:
        log.write('\nCOMMAND ' + json.dumps(command, ensure_ascii=False) + '\n')
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as child:
            try:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                raise
        if code:
            raise RuntimeError(f'Child exited with code {code}. See {log_path}; rerun the same command to resume.')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--config', type=Path)
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--train-size', type=int, default=2000)
    p.add_argument('--monitor-size', type=int, default=800)
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--min-epochs', type=int, default=8)
    p.add_argument('--warmup-epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='cpu')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError('Seeds must be distinct and nonnegative.')
    if min(args.train_size, args.threads, args.batch_size, args.patience, args.min_epochs) < 1:
        raise ValueError('Positive train size, threads, batch size, patience and minimum epochs required.')
    if args.train_size < 2 or args.monitor_size < 10 or args.image_size < 64:
        raise ValueError('train-size >=2, monitor-size >=10, image-size >=64 required.')
    if not 0 <= args.warmup_epochs < args.epochs or args.min_epochs > args.epochs:
        raise ValueError('Invalid epoch/warm-up limits.')
    root = args.project_dir.resolve()
    config = args.config or root/'al_config.json'
    if not config.is_absolute():
        config = root/config
    if not config.is_file():
        raise FileNotFoundError(f'Missing original config: {config}')
    bundle = Path(__file__).resolve().parent
    engine = bundle/'image_budget_runner'
    files = [Path(__file__), bundle/'summarize_image_2000.py', *engine.glob('*.py'), *engine.glob('galaxy_image_demo/*.py')]
    versions = {n: importlib.metadata.version(n) for n in ['torch','torchvision','numpy','h5py','matplotlib']}
    versions.update(python=sys.version, platform=platform.platform())
    # Seed membership is excluded: adding seeds reuses finished runs and extends the same campaign.
    settings = {k:v for k,v in vars(args).items() if k not in ['seeds','project_dir','config']}
    protocol = {'schema':1, 'settings':settings, 'models':MODELS, 'ensemble_members':MEMBERS,
                'project_directory':str(root), 'config':str(config), 'config_sha256':digest(config),
                'versions':versions, 'code_sha256':{str(p.relative_to(bundle)):digest(p) for p in sorted(files)},
                'test_evaluated':False, 'purpose':'model development; not an active learning experiment'}
    key = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:16]
    campaign = root/'outputs/image_2000'/('comparison_'+key)
    campaign.mkdir(parents=True,exist_ok=True)
    saved = campaign/'campaign_protocol.json'
    if saved.exists() and read_json(saved) != protocol:
        raise ValueError('Campaign protocol changed.')
    write_json(saved,protocol)
    registry_path = campaign/'runs.json'
    registry = read_json(registry_path) if registry_path.exists() else {'schema':1,'runs':{}}
    write_json(root/'outputs/latest_image_2000.json',{'campaign_directory':str(campaign), 'status':'running', 'requested_seeds':args.seeds})
    print(f'Campaign: {campaign}\nSame {args.train_size} labels per model; maximum {args.epochs} epochs; patience {args.patience}.',flush=True)
    for seed in args.seeds:
        command = [sys.executable,'-u',str(engine/'run_image_demo.py'),'--project-dir',str(root),'--config',str(config),
                   '--models',*MODELS,'--seed',str(seed)]
        for name,value in settings.items():
            command += ['--'+name.replace('_','-'),str(value)]
        stream_process(command,campaign/f'seed_{seed}.log',root)
        pointer = read_json(root/'outputs/latest_image_demo.json')
        experiment = Path(pointer['experiment_directory'])
        actual = read_json(experiment/'locked_protocol.json')
        if actual['settings']['seed'] != seed or actual['train_size'] != args.train_size or pointer['models'] != MODELS:
            raise ValueError('Latest child pointer belongs to a different run. Do not run concurrent pilots.')
        stream_process([sys.executable,'-u',str(engine/'analyze_image_ensemble.py'),'--project-dir',str(root),
                        '--experiment',str(experiment),'--members',*MEMBERS],campaign/f'seed_{seed}.log',root)
        ensemble = experiment/'ensembles'/('__'.join(MEMBERS))
        registry['runs'][str(seed)] = {'seed':seed,'experiment_directory':str(experiment),
            'report_directory':pointer['report_directory'], 'ensemble_directory':str(ensemble)}
        write_json(registry_path,registry)
        # Commit a useful summary after every completed seed.
        stream_process([sys.executable,'-u',str(bundle/'summarize_image_2000.py'),'--campaign',str(campaign)],campaign/'summary.log',root)
    write_json(root/'outputs/latest_image_2000.json',{'campaign_directory':str(campaign), 'status':'complete',
        'requested_seeds':args.seeds,'completed_seeds':sorted(map(int,registry['runs'])),
        'report':str(campaign/'image_2000_report.md')})
    print(f'\nCOMPLETE: {campaign / "image_2000_report.md"}',flush=True)


if __name__ == '__main__':
    try:
        main()
    except (Exception,KeyboardInterrupt) as exc:
        print(f'\nIMAGE 2000 STOPPED: {exc}\nRerun the same command to resume; original data and old scripts are unchanged.',file=sys.stderr)
        sys.exit(1)
