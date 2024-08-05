"""The channel model."""

from typing import overload

import functools
import math
import numpy as np
import numpy.typing as npt
from scipy.stats import uniform, norm
from itur.models import itu676

from ..util import constant
from ..util.distribution import Rayleigh, Nakagami
from .. import util


class Channel():
  """The class of wireless channel"""

  def __init__(self):
    self.rayleigh = Rayleigh()
    self.nakagami = Nakagami()

  def free_space(self, distance: float, freq: float) -> float:
    """The free space path loss model.

    Args:
      distance (float): The propagation distance.
      freq (float): The center frequency.

    Returns:
      (float): The path loss in dB
    """
    pl = 2 * util.todb(distance) + 2 * util.todb(freq) + constant.FREE_SPACE_LOSS
    pl = float(pl)
    return pl

  def shadowed_rician_fading(self, b: float, m: float, Omega: float) -> float:
    """The shadowed rician fading model.
       Reference: A new simple model for land mobile satellite channels first- and second-order statistics

    Args:
      b (float): half power of multipath component
      m (float): Nakagami parameter
      Omega (float): Average power of LOS component

    Returns:
      (float): The shadow fading in dB
    """
    A = self.rayleigh.rvs(scale=math.sqrt(b))
    Z = self.nakagami.rvs(nu=m, scale=math.sqrt(Omega))
    alpha = uniform.rvs() * 2 * constant.PI
    R = A * np.exp(1j * alpha) + Z
    return abs(R)

  @overload
  def scintillation_loss(self, epsilon: float) -> float:
    ...

  @overload
  def scintillation_loss(self, epsilon: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    ...

  def scintillation_loss(self, epsilon):
    """The scintillation loss model.

    Args:
      epsilon (float): The elevation angle in radian.

    Returns:
      (float): The scintillation loss in dB
    """
    rounded_epsilon = np.around(epsilon / constant.PI_IN_RAD, decimals=-1)
    epsilon_index = (
        rounded_epsilon /
        round(constant.ANGLE_RESOLUTION / constant.PI_IN_RAD)).astype(int)
    scint_loss = constant.SCINTILLATION_TABLE[epsilon_index]
    return scint_loss

  @overload
  def gas_attenuation(self, fc: float, epsilon: float) -> float:
    ...

  @overload
  def gas_attenuation(self, fc: float, epsilon: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    ...

  def gas_attenuation(self, fc, epsilon):
    """The gas attenuation model.
    The gas attenuation is zenith loss / sin(elevation_angle).
    The longer the distance through the troposphere (small elevation angle),
    the more severe attenuation it suffers

    Args:
      fc (float): The central frequency of the signal
      epsilon (float): The elevation angle in radian.

    Returns:
      (float): The gas attenuation in dB
    """
    return self.zenith_attenuation(fc=fc) / np.sin(epsilon)

  def zenith_attenuation(self, fc: float) -> float:
    """The zenith attenuation model.
    Zenith loss is the loss when the elevation angle is 90 degrees.

    Args:
      fc (float): The central frequency of the signal

    Returns:
      (float): The zenith attenuation in dB
    """
    zenith_att = itu676.gaseous_attenuation_slant_path(
        f=fc / 1e9,
        el=constant.PI / constant.PI_IN_RAD / 2,
        rho=constant.GROUND_WATER_VAP_DENSITY,
        P=constant.GROUND_ATMOS_PRESSURE,
        T=constant.GROUND_TEMPERATURE,
        mode='approx')

    return float(zenith_att.value)

  @functools.cache
  def cal_deterministic_loss(self, distance: float, freq: float, epsilon: float) -> float:
    """Calculate the deterministic part of the loss.

    Args:
      distance (float): The distance between sat and ue.
      freq (float): The center freq.
      epsilon (float): The elevation angle pointing from ue to sat

    Returns:
      (float): The total deterministic loss (dB)
    """
    fspl = self.free_space(distance=distance, freq=freq)
    scpl = self.scintillation_loss(epsilon)
    gpl = self.gas_attenuation(freq, epsilon)
    return fspl + scpl + gpl

  def cal_stochastic_loss(self) -> float:
    """Calculate the stochastic part of the loss.

    Args:

    Returns:
      (float): The total stochastic loss (dB)
    """
    sr_loss = self.shadowed_rician_fading(b=constant.SCATTER_COMPONENT_HALF_POWER,
                                          m=constant.NAKAGAMI_PARAMETER,
                                          Omega=constant.LOS_COMPONENT_POWER)
    return sr_loss

  def cal_total_loss(self, distance: float, freq: float, epsilon: float) -> float:
    """Calculate the total loss for sat and ue.

    Args:
      distance (float): The distance between sat and ue.
      freq (float): The center freq.
      epsilon (float): The elevation angle pointing from ue to sat

    Returns:
      (float): The total loss (dB)
    """
    det_loss = self.cal_deterministic_loss(distance=round(distance,
                                                          constant.CACHED_PRECISION),
                                           freq=round(freq,
                                                      constant.CACHED_PRECISION),
                                           epsilon=round(epsilon,
                                                         constant.CACHED_PRECISION))
    stoch_loss = self.cal_stochastic_loss()

    return det_loss + stoch_loss

  def rician_fading(self, k_db: float) -> float:
    """Rician fading.

      Args:
        k_db (float): The ratio between the power of LOS and NLOS (dB).

      Returns:
        float: The fading gain (dB).
    """
    K = util.tolinear(k_db)
    mu = np.sqrt(K / (K + 1))
    sigma = np.sqrt(1 / (2 * (K + 1)))

    h = (sigma * norm.rvs() + mu) + 1j * (sigma * norm.rvs() + mu)
    fading_gain = np.abs(h)
    return util.todb(fading_gain)

  def rayleigh_fading(self, b: float) -> float:
    """Rayleigh fading.

    Returns:
      float: The fading gain (dB).
    """
    return self.rician_fading(k_db=constant.MIN_NEG_FLOAT)
