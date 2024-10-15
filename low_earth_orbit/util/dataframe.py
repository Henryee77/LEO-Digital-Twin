"""dataframe.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import numpy.typing as npt
from . import constant


@dataclass
class RxData:
  """The rx data for each beam."""
  rsrp: float
  # sinr: float = 0
  inter_sat: bool = True

  @property
  def a3_offset(self):
    if self.inter_sat:
      return constant.A3_INTER_SAT_OFFSET

    return constant.A3_INTRA_SAT_OFFSET

  @property
  def gamma(self):
    return self.rsrp - self.a3_offset - constant.A3_HYSTERESIS


@dataclass
class AtmosphericData:
  """The Atmospheric data for whole region"""
  latitude_range: Tuple[float]
  longitude_range: Tuple[float]
  rainfall: npt.NDArray[np.float64]
  cloudtop: npt.NDArray[np.float64]

  @property
  def rainfall(self):
    return self.rainfall

  @property
  def cloudtop(self):
    return self.cloudtop


SatBeamID = Tuple[str, int]

UEservable = Dict[SatBeamID, RxData]
