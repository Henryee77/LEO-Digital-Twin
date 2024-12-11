"""Custom Distribution"""

import math
from scipy.stats import rv_continuous
from low_earth_orbit.util import util


class Rainfall_rv(rv_continuous):
  def __init__(self, r, p_0, xtol=1e-14, seed=None):
    super().__init__(a=0, xtol=xtol, seed=seed)
    self.r_i = r
    self.p_0 = p_0

  def _cdf(self, x):
    return 1 - self.p_0 * util.qfunc((math.log(x) + 0.7938 - math.log(self.r_i)) / 1.26)
