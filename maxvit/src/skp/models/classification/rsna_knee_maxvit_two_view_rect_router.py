"""Two-view fusion with rectangular MaxViT compartment inputs."""

from skp.models.classification.rsna_knee_maxvit_rect_router import Net as Rectangular
from skp.models.classification.rsna_knee_maxvit_two_view_router import Net as TwoView


class Net(TwoView, Rectangular):
    """Reuse two-view forward and tested rectangular crop/attention adaptation.

    Cooperative initialization installs rectangular attention before TwoView
    removes the unused global sequence head and adds regional feature fusion.
    Depth remains 32 for BOTH views in every plane.
    """
