"""Custom Distribution"""

import math
import numpy as np
import numpy.typing as npt
from typing import overload
from scipy.stats import rv_continuous
from scipy.stats import rayleigh, nakagami, uniform, norm
from scipy.special import hyp1f1


class Shadowed_Rician(rv_continuous):
  """The shadowed rician distribution, 
  which is used in the fading channel of LEO to ground.
  """

  # , LOS_power: float, NLOS_power: float, m: float
  def _pdf(self, x: float, LOS_power: float, NLOS_power: float, m: float) -> float:
    """The PDF of the shadowed Rician fading.

    Args:
      x (float): input
      LOS_power (float): The average power of the LOS component.
      NLOS_power (float): The average power of the multipath component.
      m (float): Nakagami parameter.


    Returns:
      float: The probability of x.
    """
    # print(args, type(args))
    # (LOS_power, NLOS_power, m) = args
    NLOS_ratio = NLOS_power * m / (NLOS_power * m + LOS_power)
    alpha = 1 / NLOS_power * (NLOS_ratio ** m)
    beta = 1 / NLOS_power
    delta = (NLOS_power ** 2 / 2 * m + NLOS_power / 2 * LOS_power) * LOS_power / 2
    print(alpha, math.exp(-beta * x), hyp1f1(m, 1, delta * x))
    return alpha * math.exp(-beta * x) * hyp1f1(m, 1, delta * x)


class Rayleigh():
  """Rayleigh Distribution"""

  def __init__(self) -> None:
    self.distribution = rayleigh

  @overload
  def rvs(self, scale: float) -> float:
    ...

  @overload
  def rvs(self, scale: float, size: int) -> npt.NDArray[np.float64]:
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
    ...

  @overload
  def rvs(self, nu: float, scale: float, size: int) -> npt.NDArray[np.float64]:
    ...

  def rvs(self, nu, scale, size=None):
    if size is None:
      return self.distribution.rvs(nu=nu, scale=scale)
    else:
      return np.array([self.distribution.rvs(nu=nu, scale=scale) for i in range(size)])
