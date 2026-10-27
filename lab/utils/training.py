import torch
import random
import os
import numpy as np
import glob

def calc_grad_norm(parameters, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        total_norm = None
    return total_norm

def sync_across_gpus(t, world_size):
    torch.distributed.barrier()
    gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]
    torch.distributed.all_gather(gather_t_tensor, t)
    return torch.cat(gather_t_tensor)

def set_seed(seed=1234, cudnn_benchmark=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = cudnn_benchmark

def create_checkpoint(cfg, model, optimizer, epoch, scheduler=None, scaler=None):
    if cfg.save_weights_only:
        checkpoint = {'model': model.state_dict()}
        return checkpoint
    checkpoint = {'model': model.state_dict(), 'epoch': epoch}
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    if scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()
    if getattr(cfg, 'save_optimizer_state', False):
        checkpoint['optimizer'] = optimizer.state_dict()
    return checkpoint

def find_checkpoint(cfg, checkpoint_path=None):
    fold_dir = f'{cfg.output_dir}/fold{cfg.fold}'
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            print(f'Warning: Specified checkpoint not found: {checkpoint_path}')
            return (None, None)
        seed = None
        if 'seed' in checkpoint_path:
            try:
                seed = int(checkpoint_path.split('seed')[-1].replace('.pth', ''))
            except (ValueError, IndexError):
                pass
        return (checkpoint_path, seed)
    if cfg.seed >= 0:
        ckpt_path = f'{fold_dir}/checkpoint_last_seed{cfg.seed}.pth'
        if os.path.exists(ckpt_path):
            print(f'Found checkpoint by seed: {ckpt_path}')
            return (ckpt_path, cfg.seed)
    pattern = f'{fold_dir}/checkpoint_last_seed*.pth'
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return (None, None)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    ckpt_path = checkpoints[0]
    try:
        seed = int(ckpt_path.split('seed')[-1].replace('.pth', ''))
    except (ValueError, IndexError):
        seed = None
    print(f'Found most recent checkpoint: {ckpt_path} (seed={seed})')
    return (ckpt_path, seed)

def load_checkpoint_for_resume(ckpt_path, model, optimizer, scheduler, scaler, device):
    print(f'Loading checkpoint from: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'epoch' not in checkpoint:
        print('\n' + '=' * 70)
        print('WARNING: Checkpoint missing epoch info, cannot properly resume')
        print('This checkpoint only contains:', list(checkpoint.keys()))
        print('=' * 70 + '\n')
    model.load_state_dict(checkpoint['model'])
    print('  - Model weights loaded')
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print('  - Optimizer state loaded')
    else:
        print('  - WARNING: No optimizer state - optimizer will start fresh!')
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler_state = checkpoint['scheduler'].copy()
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) and scheduler_state['total_steps'] != scheduler.total_steps:
            scheduler_state['total_steps'] = scheduler.total_steps
            scheduler_state['_schedule_phases'] = scheduler.state_dict()['_schedule_phases']
        scheduler.load_state_dict(scheduler_state)
        print('  - Scheduler state loaded')
    elif scheduler is not None:
        print('  - WARNING: No scheduler state - LR schedule will restart from beginning!')
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        print('  - GradScaler state loaded')
    elif scaler is not None:
        print('  - WARNING: No scaler state in checkpoint')
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        print(f'  - Resuming from epoch {start_epoch} (checkpoint was epoch {checkpoint['epoch']})')
    else:
        start_epoch = 0
        print('  - WARNING: No epoch in checkpoint, starting from epoch 0')
    return (model, optimizer, scheduler, scaler, start_epoch)

class EMA:

    def __init__(self, model, decay, use_warmup=False):
        self.decay = decay
        self.use_warmup = use_warmup
        self.num_updates = 0
        self.shadow = {key: value.detach().clone().float() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        if self.use_warmup:
            step = max(0, self.num_updates - 1)
            decay = min(self.decay, 1 - (1 + step) ** (-2 / 3)) if step else 0.0
            floating = [(self.shadow[key], value.detach().float()) for key, value in model.state_dict().items() if value.dtype.is_floating_point]
            torch._foreach_lerp_([pair[0] for pair in floating], [pair[1] for pair in floating], 1 - decay)
            for key, value in model.state_dict().items():
                if not value.dtype.is_floating_point:
                    self.shadow[key] = value.detach().clone().float()
            self.num_updates += 1
            return
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(value.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[key] = value.detach().clone().float()

    def state_dict(self, model):
        return {key: self.shadow[key].to(value.dtype) for key, value in model.state_dict().items()}

    def apply(self, model):
        from contextlib import contextmanager

        @contextmanager
        def swapped():
            original = {key: value.detach().clone() for key, value in model.state_dict().items()}
            model.load_state_dict(self.state_dict(model))
            try:
                yield
            finally:
                model.load_state_dict(original)
        return swapped()
