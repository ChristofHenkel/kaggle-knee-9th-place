from lightning.pytorch.utilities import rank_zero_only


@rank_zero_only
def _print_rank_zero(msg: str) -> None:
    print(msg)
