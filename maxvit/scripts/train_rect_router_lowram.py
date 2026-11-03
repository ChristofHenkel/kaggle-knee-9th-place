"""Launch rectangular router with explicit, logged low-RAM loader settings."""

from torch.utils.data import DataLoader
from skp import train
from skp.tasks import utils


def low_ram_loader(*args, **kwargs):
    if kwargs.get("num_workers") != 1 or kwargs.get("persistent_workers"):
        raise ValueError("Expected one nonpersistent worker from candidate config")
    kwargs["prefetch_factor"] = 1
    print(
        "Rectangular router loader: workers=1, prefetch=1, persistent=False", flush=True
    )
    return DataLoader(*args, **kwargs)


if __name__ == "__main__":
    utils.DataLoader = low_ram_loader
    train.main()
