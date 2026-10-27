import argparse
from collections import defaultdict
import importlib
import json
from pathlib import Path

import torch

from .artifacts import sha256
from .config import setup_sys_path


def export(config, run_root, output):
    setup_sys_path()
    cfg = importlib.import_module(config).cfg
    run_root, output = Path(run_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fullfit = config in ('cfg_1_91', 'cfg_8_31')
    variant = 'ema' if config in ('cfg_8_29', 'cfg_8_31', 'cfg_8_32') else 'model'
    members = []
    for index in range(1 if config == 'cfg_18_7' else 5):
        folder = run_root / f'repeat{index + 1}' / 'fold-1' if fullfit else run_root / f'fold{index}'
        files = list(folder.glob('checkpoint_last_seed*.pth'))
        if len(files) != 1:
            raise ValueError(f'Expected one final checkpoint in {folder}; found {len(files)}')
        source = files[0]
        with torch.serialization.safe_globals([defaultdict, list]):
            checkpoint = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        if checkpoint['epoch'] + 1 != cfg.epochs:
            raise ValueError(f'Expected final epoch {cfg.epochs}: {source}')
        state = {key.removeprefix('module.'): value for key, value in checkpoint[variant].items()}
        path = output / ('model.pth' if config == 'cfg_18_7' else f'member{index}.pth')
        torch.save({'model': state, 'epoch': checkpoint['epoch']}, path)
        members.append({'file': path.name, 'sha256': sha256(path), 'epoch': cfg.epochs,
                        'seed': int(source.stem.split('seed')[-1])})
    manifest = {'config': config, 'variant': variant,
                'member_kind': 'independent_fullfit_seed' if fullfit else 'cv_fold',
                'aggregation': 'probability_mean', 'members': members}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=['cfg_1_85', 'cfg_1_91', 'cfg_8_29', 'cfg_8_31', 'cfg_8_32', 'cfg_18_7'], required=True)
    parser.add_argument('--run-root')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output or (root / 'outputs/cfg_18_7' if args.config == 'cfg_18_7' else root / 'outputs/export' / args.config)
    export(args.config, args.run_root or root / 'outputs' / args.config, output)


if __name__ == '__main__':
    main()
