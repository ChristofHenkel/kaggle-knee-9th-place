import importlib
import sys
from pathlib import Path


def setup_sys_path():
    """Add standard module directories to sys.path for dynamic imports."""
    lab = Path(__file__).resolve().parents[1]
    for directory in ("", "configs", "models", "datasets", "postprocess", "metrics"):
        path = str(lab / directory)
        if path not in sys.path:
            sys.path.append(path)


def apply_cli_overrides(cfg, other_args):
    """Parse key-value pairs from other_args and override cfg fields."""
    if len(other_args) <= 1:
        return
    arguments = other_args[1:] if other_args[0] == '--' else other_args
    pairs = {k.replace('-', ''): v for k, v in zip(arguments[::2], arguments[1::2])}
    for key, val in pairs.items():
        if key in cfg.__dict__:
            cfg_type = type(cfg.__dict__[key])
            print(f'overwriting cfg.{key}: {cfg.__dict__[key]} -> {val}')
            if val == 'None':
                cfg.__dict__[key] = None
            elif cfg_type == bool:
                cfg.__dict__[key] = val == 'True'
            elif cfg_type == type(None):
                cfg.__dict__[key] = val
            elif cfg_type == list:
                cfg.__dict__[key] = val.split(',')
            else:
                cfg.__dict__[key] = cfg_type(val)
        else:
            print(f'unknown key: {key}')
            cfg.__dict__[key] = val


def load_dynamic_modules(cfg):
    """Import dataset, postprocess, and metric callables into cfg."""
    cfg.CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
    cfg.tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
    cfg.val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
    cfg.batch_to_device = importlib.import_module(cfg.dataset).batch_to_device
    cfg.post_process_pipeline = importlib.import_module(cfg.post_process_pipeline).post_process_pipeline
    cfg.calc_metric = importlib.import_module(cfg.metric).calc_metric
    dataset_module = importlib.import_module(cfg.dataset)
    if hasattr(dataset_module, "prepare_data"):
        cfg.prepare_data = dataset_module.prepare_data
    metric_module = importlib.import_module(cfg.metric)
    if hasattr(metric_module, "summarize_epoch"):
        cfg.summarize_epoch = metric_module.summarize_epoch
    if hasattr(metric_module, "resume_history"):
        cfg.resume_history = metric_module.resume_history
