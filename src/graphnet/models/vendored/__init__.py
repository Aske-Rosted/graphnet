"""Model classes vendored from other GraphNeT forks, for deployment only.

`bundle`: the classes Maxwell Nakos' production bundle models need beyond
upstream GraphNeT (see README.md). They are found by `ModelConfig` like any
other GraphNeT class, so the models' `model_config.yml` files load unchanged.
"""

from .bundle import (
    FocalCrossEntropyLoss,
    IceCubeBundleAdvanced,
    LateralSpreadReconstruction,
    NodesAsPulsesBundle,
    RMSEAdjustedLoss,
)
