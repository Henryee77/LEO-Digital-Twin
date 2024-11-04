"""Custom Distribution"""

import math
import numpy as np
import numpy.typing as npt
from typing import overload
from scipy.stats import rv_continuous
from scipy.stats import rayleigh, nakagami, uniform, norm
from scipy.special import hyp1f1
from low_earth_orbit.util import util


class Rayleigh():
  """Rayleigh Distribution"""

  def __init__(self) -> None:
    self.distribution = rayleigh

  @overload
  def rvs(self, scale: float) -> float:
    """Generate the Rayleigh random number.

    Args:
        scale (float): The scale parameter (σ) of the rayleigh distribution.

    Returns:
        float: The generated number.
    """
    ...

  @overload
  def rvs(self, scale: float, size: int) -> npt.NDArray[np.float64]:
    """Generate the Rayleigh random number.

    Args:
        scale (float): The scale parameter (σ) of the rayleigh distribution.
        size (int): The total number of generated numbers.

    Returns:
        npt.NDArray[np.float64]: The array of generated number.
    """
    ...

  def rvs(self, scale, size=None):
    if size is None:
      return self.distribution.rvs(scale=scale)
    else:
      return np.array([self.distribution.rvs(scale=scale) for i in range(size)])


class Nakagami():
  """Nakagami Distribution"""

  def __init__(self) -> None:
    self.distribution = nakagami

  @overload
  def rvs(self, nu: float, scale: float) -> float:
    """Generate the Nakagami random number.

    Args:
        nu (float): The Nakagami parameter (m).
        scale (float): The scale parameter (Ω) of the rayleigh distribution.

    Returns:
        float: The generated number.
    """
    ...

  @overload
  def rvs(self, nu: float, scale: float, size: int) -> npt.NDArray[np.float64]:
    """Generate the Nakagami random number.

    Args:
        nu (float): The Nakagami parameter (m).
        scale (float): The scale parameter (Ω) of the rayleigh distribution.
        size (int): The total number of generated numbers.

    Returns:
        npt.NDArray[np.float64]: The array of generated number.
    """
    ...

  def rvs(self, nu, scale, size=None):
    if size is None:
      return self.distribution.rvs(nu=nu, scale=scale)
    else:
      return np.array([self.distribution.rvs(nu=nu, scale=scale) for i in range(size)])


class Rainfall_rv(rv_continuous):
  def __init__(self, r, p_0, xtol=1e-14, seed=None):
    super().__init__(a=0, xtol=xtol, seed=seed)
    self.r_i = r
    self.p_0 = p_0

  def _cdf(self, x):
    return 1 - self.p_0 * util.qfunc((math.log(x) + 0.7938 - math.log(self.r_i)) / 1.26)
