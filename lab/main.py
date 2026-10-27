import numpy as np
import sys
from tqdm import tqdm
import gc
import argparse
import torch
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from collections import defaultdict
from contextlib import suppress
from functools import partial
import os
import time
import tempfile
import warnings
warnings.filterwarnings('ignore', message='.*TensorRT-LLM is not installed.*')
from utils.training import sync_across_gpus, set_seed, create_checkpoint, find_checkpoint, load_checkpoint_for_resume, calc_grad_norm, EMA
from utils.data import get_data, get_dataset, get_dataloader
from utils.model import get_model, convert_sync_batchnorm
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from utils.config import apply_cli_overrides, load_dynamic_modules, setup_sys_path
from utils.artifacts import resolve_artifacts
from copy import copy
setup_sys_path()
try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    print('no cv2 installed, running without')

def setup_distributed(cfg):
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1000000)
    if getattr(cfg, 'seed_fold_offset', False):
        cfg.seed += cfg.fold
    print('seed', cfg.seed)
    cfg.local_rank = 0
    cfg.world_size = 1
    cfg.rank = 0
    cfg.distributed = False
    if isinstance(cfg.amp_dtype, str):
        cfg.amp_dtype = getattr(torch, cfg.amp_dtype)
    if isinstance(cfg.eval_amp_dtype, str):
        cfg.eval_amp_dtype = getattr(torch, cfg.eval_amp_dtype)
    if 'WORLD_SIZE' in os.environ:
        cfg.distributed = int(os.environ['WORLD_SIZE']) > 1
    if cfg.distributed:
        cfg.local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device('cuda:%d' % cfg.local_rank)
        cfg.device = device
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        cfg.world_size = torch.distributed.get_world_size()
        cfg.rank = torch.distributed.get_rank()
        print(f'Process {cfg.rank}, total {cfg.world_size}, local rank {cfg.local_rank}.')
        cfg.group = torch.distributed.new_group(np.arange(cfg.world_size))
        cfg.seed = int(sync_across_gpus(torch.Tensor([cfg.seed]).to(device), cfg.world_size).detach().cpu().numpy()[0])
        print(f'LOCAL_RANK {cfg.local_rank}, device {device}, seed {cfg.seed}')
    else:
        device = torch.device('cuda:%d' % cfg.gpu)
        cfg.device = device
    set_seed(cfg.seed, cudnn_benchmark=getattr(cfg, 'cudnn_benchmark', True))
    if hasattr(cfg, 'float32_matmul_precision'):
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    return device

def setup_amp(cfg, device):
    amp_autocast = suppress
    loss_scaler = None
    if cfg.amp_dtype is not None:
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=cfg.amp_dtype)
        if device.type in ('cuda',) and cfg.amp_dtype == torch.float16:
            loss_scaler = GradScaler(device.type)
    return (amp_autocast, loss_scaler)

def run_eval_dispatch(model, dataloader, cfg, amp_autocast=suppress, **kwargs):
    if cfg.distributed and cfg.eval_ddp:
        return run_eval(model, dataloader, cfg, amp_autocast=amp_autocast, **kwargs)
    elif cfg.local_rank == 0:
        return run_eval(model, dataloader, cfg, amp_autocast=amp_autocast, **kwargs)
    return 0

def save_checkpoint_if_needed(cfg, model, optimizer, epoch, scheduler, loss_scaler):
    if cfg.local_rank == 0 and cfg.epochs > 0 and cfg.save_checkpoint:
        checkpoint = create_checkpoint(cfg, model, optimizer, epoch, scheduler=scheduler, scaler=loss_scaler)
        if hasattr(cfg, '_ema'):
            checkpoint['ema'] = cfg._ema.state_dict(model)
            if getattr(cfg, 'ema_warmup', False):
                checkpoint['ema_updates'] = cfg._ema.num_updates
        path = f'{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth'
        if cfg.resumable:
            with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), suffix='.pth', delete=False) as file:
                temporary = file.name
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
        else:
            torch.save(checkpoint, path)
        keep_every = getattr(cfg, 'checkpoint_keep_every', 0)
        if keep_every and (epoch + 1) % keep_every == 0:
            snapshot = f'{cfg.output_dir}/fold{cfg.fold}/checkpoint_epoch{epoch + 1}_seed{cfg.seed}.pth'
            torch.save(checkpoint, snapshot)

def mean_train_losses(losses, cfg, device):
    means = {key: float(np.mean(values)) for key, values in losses.items()}
    if cfg.distributed and getattr(cfg, 'reduce_train_losses', False):
        for key, values in losses.items():
            totals = torch.tensor([sum(values), len(values)], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(totals)
            means[key] = (totals[0] / totals[1]).item()
    return means

def train_one_epoch(epoch, model, train_dataloader, val_dataloader, optimizer, scheduler, loss_scaler, amp_autocast, cfg, device, iteration_counter):
    if getattr(cfg, 'reseed_each_epoch', True):
        set_seed(cfg.seed + epoch + cfg.local_rank, cudnn_benchmark=getattr(cfg, 'cudnn_benchmark', True))
    if getattr(cfg, 'epoch_accumulation', False):
        optimizer.zero_grad(set_to_none=True)
    cfg.curr_epoch = epoch
    if cfg.local_rank == 0:
        print('EPOCH:', epoch)
    if hasattr(train_dataloader.sampler, 'set_epoch'):
        train_dataloader.sampler.set_epoch(epoch)
    progress_bar = tqdm(range(len(train_dataloader)), disable=cfg.disable_tqdm)
    tr_it = iter(train_dataloader)
    losses = []
    epoch_losses = defaultdict(list)
    interval_losses = defaultdict(list)
    evals_per_epoch = getattr(cfg, 'evals_per_epoch', 1)
    eval_batches = {(part * len(train_dataloader) + evals_per_epoch - 1) // evals_per_epoch for part in range(1, evals_per_epoch)}
    total_grad_norm = None
    total_grad_norm_after_clip = None
    gc.collect()
    i = iteration_counter
    for itr in progress_bar:
        i += 1
        cfg.curr_step += cfg.batch_size * cfg.world_size
        try:
            data = next(tr_it)
        except Exception as e:
            print(e)
            print('DATA FETCH ERROR')
        model.train()
        torch.set_grad_enabled(True)
        batch = cfg.batch_to_device(data, device)
        with amp_autocast():
            output_dict = model(batch)
        loss = output_dict['loss']
        losses.append(loss.item())
        for key, value in output_dict.items():
            if 'loss' in key:
                epoch_losses[key].append(value.item())
                if evals_per_epoch > 1:
                    interval_losses[key].append(value.item())
        if cfg.grad_accumulation > 1:
            loss /= cfg.grad_accumulation
        if loss_scaler is not None:
            loss_scaler.scale(loss).backward()
        else:
            loss.backward()
        accumulation_step = itr + 1 if getattr(cfg, 'epoch_accumulation', False) else i
        if accumulation_step % cfg.grad_accumulation == 0:
            if loss_scaler is not None and (cfg.track_grad_norm or cfg.clip_grad > 0):
                loss_scaler.unscale_(optimizer)
            if cfg.track_grad_norm:
                total_grad_norm = calc_grad_norm(model.parameters(), cfg.grad_norm_type)
            if cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            if cfg.track_grad_norm:
                total_grad_norm_after_clip = calc_grad_norm(model.parameters(), cfg.grad_norm_type)
            if loss_scaler is not None:
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if getattr(cfg, 'schedule_on_optimizer_step', False) and scheduler is not None:
                scheduler.step()
            if hasattr(cfg, '_ema'):
                cfg._ema.update(model)
        if cfg.distributed:
            torch.cuda.synchronize()
        if scheduler is not None and (not getattr(cfg, 'schedule_on_optimizer_step', False)):
            scheduler.step()
        if cfg.local_rank == 0 and cfg.curr_step % cfg.batch_size == 0:
            progress_bar.set_description(f'loss: {np.mean(losses[-10:]):.4f}')
        if cfg.val and itr + 1 in eval_batches:
            interval_means = mean_train_losses(interval_losses, cfg, device)
            progress = epoch + (itr + 1) / len(train_dataloader)
            if cfg.local_rank == 0:
                print(f'Epoch {progress:.4f}: train_loss={interval_means['loss']:.4f} lr={optimizer.param_groups[0]['lr']:.8g}', flush=True)
            run_eval_dispatch(model, val_dataloader, cfg, amp_autocast=amp_autocast, pre='val', curr_epoch=epoch)
            interval_losses.clear()
        elif cfg.eval_steps != 0:
            if i % cfg.eval_steps == 0:
                run_eval_dispatch(model, val_dataloader, cfg, pre='val', curr_epoch=epoch)
    train_losses = mean_train_losses(epoch_losses, cfg, device)
    logged_losses = mean_train_losses(interval_losses, cfg, device) if evals_per_epoch > 1 else train_losses
    print(f'Mean train_loss {np.mean(losses):.4f}')
    cfg.last_train_losses = train_losses
    return i

def run_eval(model, val_dataloader, cfg, pre='val', curr_epoch=0, amp_autocast=suppress):
    model.eval()
    torch.set_grad_enabled(False)
    log_step = cfg.curr_step if getattr(cfg, 'evals_per_epoch', 1) > 1 else curr_epoch
    val_data = defaultdict(list)
    val_score = 0
    if cfg.eval_amp_dtype == torch.float16 or cfg.eval_amp_dtype is None:
        if cfg.eval_half_model:
            model.half()
    for ind_, data in enumerate(tqdm(val_dataloader, disable=(cfg.local_rank != 0) | cfg.disable_tqdm)):
        batch = cfg.batch_to_device(data, cfg.device)
        if cfg.eval_amp_dtype == torch.float16:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(batch)
        elif cfg.eval_amp_dtype is None:
            output = model(batch)
        else:
            with amp_autocast():
                output = model(batch)
        if cfg.local_rank == 0 and cfg.calc_metric and ((curr_epoch + 1) % cfg.calc_metric_epochs == 0):
            pass
        if not getattr(run_eval, '_saved_images', False) and cfg.save_first_batch_preds:
            run_eval._saved_images = True
        for key, val in output.items():
            val_data[key] += [output[key]]
    for key, val in output.items():
        value = val_data[key]
        if isinstance(value[0], list):
            val_data[key] = [item for sublist in value for item in sublist]
        elif len(value[0].shape) == 0:
            val_data[key] = torch.stack(value)
        else:
            val_data[key] = torch.cat(value, dim=0)
    if cfg.distributed and cfg.eval_ddp:
        for key, val in output.items():
            val_data[key] = sync_across_gpus(val_data[key], cfg.world_size)
    if cfg.local_rank == 0:
        if cfg.distributed and cfg.eval_ddp:
            for k, v in val_data.items():
                val_data[k] = v[:len(val_dataloader.dataset)]
        if cfg.save_val_data:
            torch.save(val_data, f'{cfg.output_dir}/fold{cfg.fold}/{pre}_data_seed{cfg.seed}.pth')
            if getattr(cfg, 'save_val_snapshots', False):
                torch.save(val_data, f'{cfg.output_dir}/fold{cfg.fold}/{pre}_data_step{cfg.curr_step}_seed{cfg.seed}.pth')
    loss_names = [key for key in output if 'loss' in key]
    loss_names += [key for key in output if 'score' in key]
    for k in loss_names:
        if cfg.local_rank == 0 and k in val_data:
            losses = val_data[k].cpu().numpy()
            loss = np.mean(losses)
            print(f'Mean {pre}_{k}', loss)
    if cfg.local_rank == 0 and cfg.calc_metric and ((curr_epoch + 1) % cfg.calc_metric_epochs == 0):
        val_df = val_dataloader.dataset.df
        pp_out = cfg.post_process_pipeline(cfg, val_data, val_df)
        del val_data
        gc.collect()
        val_score = cfg.calc_metric(cfg, pp_out, val_df, pre)
        if not isinstance(val_score, dict):
            val_score = {'score': val_score}
        for k, v in val_score.items():
            print(f'{pre}_{k}: {v:.3f}')
    if cfg.distributed:
        torch.distributed.barrier()
    return val_score

def evaluate_saved(cfg, path):
    cfg.rank = cfg.local_rank = 0
    cfg.world_size = 1
    cfg.distributed = False
    cfg.curr_step = 0
    cfg.evaluation_source = path
    if cfg.seed < 0:
        cfg.seed = int(np.random.randint(1000000))
    print('seed', cfg.seed)
    _, val_df, _ = get_data(cfg)
    with torch.serialization.safe_globals([defaultdict, list]):
        predictions = torch.load(path, map_location='cpu', weights_only=True)
    status = 'FAILED'
    processed = cfg.post_process_pipeline(cfg, predictions, val_df)
    score = cfg.calc_metric(cfg, processed, val_df, 'val')
    status = 'FINISHED'
    return score

def train(cfg, checkpoint_path=None):
    device = setup_distributed(cfg)
    ckpt_path = None
    ckpt_seed = None
    resuming = False
    if getattr(cfg, 'resumable', False):
        ckpt_path, ckpt_seed = find_checkpoint(cfg, checkpoint_path)
        if ckpt_path is not None:
            resuming = True
    train_df, val_df, test_df = get_data(cfg)
    train_dataset = get_dataset(train_df, cfg, mode='train')
    train_dataloader = get_dataloader(train_dataset, cfg, mode='train')
    val_dataset = get_dataset(val_df, cfg, mode='val')
    val_dataloader = get_dataloader(val_dataset, cfg, mode='val')
    if cfg.test:
        test_dataset = get_dataset(test_df, cfg, mode='test')
        test_dataloader = get_dataloader(test_dataset, cfg, mode='test')
    if cfg.train_val:
        train_val_dataset = get_dataset(train_df, cfg, mode='val')
        train_val_dataloader = get_dataloader(train_val_dataset, cfg, 'val')
    model = get_model(cfg, train_dataset)
    model.to(device)
    if cfg.distributed:
        if cfg.syncbn:
            model = convert_sync_batchnorm(model, cfg)
        model = NativeDDP(model, device_ids=[cfg.local_rank], find_unused_parameters=cfg.find_unused_parameters)
    total_steps = len(train_dataset)
    if train_dataloader.sampler is not None:
        if 'WeightedRandomSampler' in str(train_dataloader.sampler.__class__):
            total_steps = train_dataloader.sampler.num_samples
    optimizer = get_optimizer(model, cfg)
    if cfg.schedule in ('parag_cosine', 'cosine_floor', 'ian_cosine'):
        cfg.optimizer_steps_per_epoch = len(train_dataloader) // cfg.grad_accumulation
    elif cfg.schedule == 'onecycle':
        cfg.optimizer_steps_per_epoch = len(train_dataloader)
        if cfg.schedule_on_optimizer_step:
            cfg.optimizer_steps_per_epoch //= cfg.grad_accumulation
    scheduler = get_scheduler(cfg, optimizer, total_steps)
    amp_autocast, loss_scaler = setup_amp(cfg, device)
    start_epoch = 0
    cfg.curr_step = 0
    if resuming:
        try:
            model, optimizer, scheduler, loss_scaler, start_epoch = load_checkpoint_for_resume(ckpt_path, model, optimizer, scheduler, loss_scaler, device)
        except Exception as e:
            print(f'Error loading checkpoint: {e}')
            print('Starting training from scratch')
            resuming = False
            start_epoch = 0
            cfg.curr_step = 0
        else:
            if ckpt_seed is not None and ckpt_seed != cfg.seed:
                print(f'Overriding seed from checkpoint: {cfg.seed} -> {ckpt_seed}')
                cfg.seed = ckpt_seed
                set_seed(cfg.seed, cudnn_benchmark=getattr(cfg, 'cudnn_benchmark', True))
                if hasattr(train_dataloader.sampler, 'seed'):
                    train_dataloader.sampler.seed = cfg.seed
            steps_per_epoch = len(train_dataloader)
            samples_per_epoch = steps_per_epoch * cfg.batch_size * cfg.world_size
            cfg.curr_step = start_epoch * samples_per_epoch
            print(f'Resuming at curr_step={cfg.curr_step} (after {start_epoch} completed epochs)')
            if cfg.local_rank == 0 and hasattr(cfg, 'resume_history'):
                cfg.resume_history(cfg, start_epoch)
    if hasattr(cfg, 'ema_decay'):
        cfg._ema = EMA(model, cfg.ema_decay, use_warmup=getattr(cfg, 'ema_warmup', False))
        if resuming:
            ema_checkpoint = torch.load(ckpt_path, map_location=device)
            cfg._ema.shadow = ema_checkpoint['ema']
            cfg._ema.num_updates = ema_checkpoint.get('ema_updates', 0)
    status = 'FAILED'
    iteration_counter = start_epoch * len(train_dataloader)
    optimizer.zero_grad()
    val_score = 0
    for epoch in range(start_epoch, cfg.epochs):
        cfg.epoch_started = time.monotonic()
        if cfg.train:
            iteration_counter = train_one_epoch(epoch, model, train_dataloader, val_dataloader, optimizer, scheduler, loss_scaler, amp_autocast, cfg, device, iteration_counter)
        if cfg.distributed:
            torch.cuda.synchronize()
        if cfg.force_fp16_weights:
            model = model.half()
        if cfg.val:
            if (epoch + 1) % cfg.eval_epochs == 0 or epoch + 1 == cfg.epochs:
                if hasattr(cfg, '_ema'):
                    raw_score = None
                    if not getattr(cfg, 'ema_validate_only', False):
                        raw_score = run_eval_dispatch(model, val_dataloader, cfg, amp_autocast=amp_autocast, pre='val_raw', curr_epoch=epoch)
                    with cfg._ema.apply(model):
                        ema_score = run_eval_dispatch(model, val_dataloader, cfg, amp_autocast=amp_autocast, pre='val_ema', curr_epoch=epoch)
                    if cfg.train and hasattr(cfg, 'summarize_epoch') and (cfg.local_rank == 0):
                        val_score = cfg.summarize_epoch(cfg, raw_score, ema_score, model, epoch)
                    else:
                        val_score = raw_score
                else:
                    val_score = run_eval_dispatch(model, val_dataloader, cfg, amp_autocast=amp_autocast, pre='val', curr_epoch=epoch)
                    if cfg.train and hasattr(cfg, 'summarize_epoch') and (cfg.local_rank == 0):
                        val_score = cfg.summarize_epoch(cfg, val_score, None, model, epoch)
            else:
                val_score = 0
        if cfg.train_val:
            if (epoch + 1) % cfg.eval_train_epochs == 0 or epoch + 1 == cfg.epochs:
                run_eval_dispatch(model, train_val_dataloader, cfg, pre=cfg.pre_train_val, curr_epoch=epoch)
        if cfg.distributed:
            torch.distributed.barrier()
        if not cfg.save_only_last_ckpt:
            save_checkpoint_if_needed(cfg, model, optimizer, epoch, scheduler, loss_scaler)
    if cfg.epochs > 0:
        save_checkpoint_if_needed(cfg, model, optimizer, cfg.epochs - 1, scheduler, loss_scaler)
    if cfg.test:
        run_eval_dispatch(model, test_dataloader, cfg, pre='test', curr_epoch=max(cfg.epochs - 1, 0))
    status = 'FINISHED'
    return val_score
if __name__ == '__main__':
    import importlib
    parser = argparse.ArgumentParser(description='Training with optional checkpoint resumption')
    parser.add_argument('-C', '--config', help='config filename', required=True)
    parser.add_argument('-G', '--gpu_id', default='', help='GPU ID')
    parser.add_argument('-D', '--debug', action='store_true', help='debugging True/ False')
    parser.add_argument('-I', '--inference', action='store_true', help='inference True/ False')
    parser.add_argument('--checkpoint', default=None, help='Explicit path to checkpoint file (only used if cfg.resumable=True)')
    parser.add_argument('--eval-data', default=None, help='Evaluate saved output tensors on CPU without model inference')
    parser.add_argument('--predict-bundle', default=None)
    parser.add_argument('--competition', default=None)
    parser.add_argument('--prediction-output', default=None)
    parser_args, other_args = parser.parse_known_args(sys.argv[1:])
    cfg = copy(importlib.import_module(parser_args.config).cfg)
    if parser_args.gpu_id != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(parser_args.gpu_id)
    if parser_args.debug:
        print('debug mode')
        cfg.train_sub_epochs = 1
        cfg.save_checkpoint = False
    if parser_args.inference:
        cfg.train = False
        cfg.epochs = 1
        cfg.save_checkpoint = False
    apply_cli_overrides(cfg, other_args)
    os.makedirs(str(cfg.output_dir + f'/fold{cfg.fold}/'), exist_ok=True)
    resolve_artifacts(cfg)
    load_dynamic_modules(cfg)
    if parser_args.predict_bundle is not None:
        from utils.inference_knee import predict
        result = predict(cfg.name, parser_args.predict_bundle, parser_args.competition, parser_args.prediction_output)
    elif parser_args.eval_data is not None:
        result = evaluate_saved(cfg, parser_args.eval_data)
    else:
        result = train(cfg, checkpoint_path=parser_args.checkpoint)
    print(result)
