import importlib
import torch


def convert_sync_batchnorm(model, cfg):
    process_group = None
    group_size = getattr(cfg, 'syncbn_group_size', cfg.world_size)
    if group_size != cfg.world_size:
        for start in range(0, cfg.world_size, group_size):
            ranks = list(range(start, start + group_size))
            group = torch.distributed.new_group(ranks)
            if cfg.rank in ranks:
                process_group = group
                print(f'SyncBatchNorm rank {cfg.rank}: ranks {ranks}', flush=True)
    if getattr(cfg, 'syncbn_timm', False):
        from timm.layers import convert_sync_batchnorm
        return convert_sync_batchnorm(model, process_group=process_group)
    return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=process_group)

# FP16 safe range (leaving some margin from actual max of 65504)
FP16_MAX = 65000.0

def safe_fp16_inference(model):
    """Prepare bfloat16-trained model for float16 inference"""
    fp16_max = 65504.0

    # 1. Clamp weights
    with torch.no_grad():
        for param in model.parameters():
            param.data.clamp_(-fp16_max, fp16_max)

    # 2. Convert to half
    model = model.half()

    # 3. Add safety hooks for intermediate activations
    def nan_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            return torch.nan_to_num(output, nan=0.0, posinf=fp16_max, neginf=-fp16_max)
        return output

    for module in model.modules():
        module.register_forward_hook(nan_hook)

    return model

def get_model(cfg, ds):
    Net = importlib.import_module(cfg.model).Net
    net = Net(cfg)
    if cfg.pretrained_weights is not None:
        if isinstance(cfg.pretrained_weights, list):
            cfg.pretrained_weights = cfg.pretrained_weights[cfg.fold]
        print(f'{cfg.local_rank}: loading weights from',cfg.pretrained_weights)
        state_dict = torch.load(cfg.pretrained_weights, map_location='cpu')
        if "model" in state_dict.keys():
            state_dict = state_dict['model']
        state_dict = {key.replace('module.',''):val for key,val in state_dict.items()}
        if cfg.pop_weights is not None:
            print(f'popping {cfg.pop_weights}')
            to_pop = []
            for key in state_dict:
                for item in cfg.pop_weights:
                    if item in key:
                        to_pop += [key]
            for key in to_pop:
                print(f'popping {key}')
                state_dict.pop(key)

        # Clamp weights to fp16-safe range if configured
        if getattr(cfg, 'clamp_weights_fp16', False):
            print(f'{cfg.local_rank}: clamping weights to fp16-safe range [-{FP16_MAX}, {FP16_MAX}]')
            clamped_count = 0
            for key, val in state_dict.items():
                if val.is_floating_point():
                    out_of_range = (val.abs() > FP16_MAX).sum().item()
                    if out_of_range > 0:
                        clamped_count += out_of_range
                        state_dict[key] = val.clamp(-FP16_MAX, FP16_MAX)
            if clamped_count > 0:
                print(f'{cfg.local_rank}: clamped {clamped_count} values that exceeded fp16 range')

        net.load_state_dict(state_dict, strict=cfg.pretrained_weights_strict)
        print(f'{cfg.local_rank}: weights loaded from',cfg.pretrained_weights)

    return net
