from copy import copy
from pathlib import Path
from cfg_18_7 import cfg as parent_cfg
cfg = copy(parent_cfg)
root = Path(__file__).resolve().parents[2]
cfg.name = Path(__file__).stem
cfg.parent = 'cfg_18_7'
cfg.output_dir = str(root / 'outputs' / cfg.name)
cfg.dataset = 'ds_moaks_competition'
cfg.metric = 'metric_moaks_competition'
cfg.train_df = str(root / 'assets/data/competition_series_moaks.csv')
cfg.cache_dir = str(root / 'assets/data/cache/cfg_1_85')
cfg.pretrained_weights = str(root / 'outputs/cfg_18_7/model.pth')
cfg.encoder_3d_pretrained = False
cfg.train = False
cfg.val = True
cfg.epochs = 1
cfg.aug = False
cfg.batch_size = 16
cfg.batch_size_val = 16
cfg.drop_last = False
cfg.save_checkpoint = False
cfg.num_workers = 8
cfg.eval_workers = 8
