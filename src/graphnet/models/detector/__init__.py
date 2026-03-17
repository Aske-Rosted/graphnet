"""Detector-specific modules, for data ingestion and standardisation."""

from .icecube import (
    IceCube86,
    IceCubeDeepCore,
    IceCubeUpgrade,
    IceCubeBundleAdvanced,
)
from .detector import Detector
from .liquido import LiquidO_v1
from .prometheus import ORCA150
