"""dataframe.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from . import constant


@dataclass
class RxData:
  """The rx date for each beam."""
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


SatBeamID = Tuple[str, int]

UEservable = Dict[SatBeamID, RxData]
