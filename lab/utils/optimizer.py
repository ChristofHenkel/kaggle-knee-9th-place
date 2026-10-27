from torch import nn, optim
from torch.optim import AdamW


def get_optimizer(model, cfg):

    params = model.parameters()
    parameter_model = model.module if hasattr(model, "module") else model
    if hasattr(parameter_model, "optimizer_parameters"):
        params = parameter_model.optimizer_parameters()

    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "AdamW_plus":
        paras = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        params = [{"params": [param for name, param in paras if (not any(nd in name for nd in no_decay))],
                   "lr": cfg.lr,
                   "weight_decay":cfg.weight_decay},
                  {"params": [param for name, param in paras if (any(nd in name for nd in no_decay))],
                   "lr": cfg.lr,
                   "weight_decay":0.},
                 ]
        optimizer = AdamW(params, lr=cfg.lr)


    elif cfg.optimizer == "AdamW":
        optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.sgd_momentum,
            nesterov=cfg.sgd_nesterov,
            weight_decay=cfg.weight_decay,
        )

    return optimizer
