"""The channel model."""

from typing import overload

import functools
import numpy as np
import numpy.typing as npt
from itur.models import itu676

from ..util import constant
from .. import util


def free_space(distance: float, freq: float) -> float:
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


@overload
def shadow_fading(epsilon: float) -> float:
  ...


@overload
def shadow_fading(epsilon: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
  ...


def shadow_fading(epsilon):
  """The shadow fading model.

  Args:
    epsilon (float): The elevation angle in radian.

  Returns:
    (float): The shadow fading in dB
  """

  rounded_epsilon = np.around(epsilon / constant.PI_IN_RAD, decimals=-1)
  stdv_table = np.array(constant.SF_STDV_LIST[constant.SF_MODEL])
  epsilon_index = (
      rounded_epsilon /
      round(constant.ANGLE_RESOLUTION / constant.PI_IN_RAD)).astype(int)
  sf_stdv = stdv_table[epsilon_index]
  sf_loss = np.abs(np.random.normal(loc=0, scale=sf_stdv))
  return sf_loss


@overload
def scintillation_loss(epsilon: float) -> float:
  ...


@overload
def scintillation_loss(
        epsilon: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
  ...


def scintillation_loss(epsilon):
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
def gas_attenuation(fc: float, epsilon: float) -> float:
  ...


@overload
def gas_attenuation(
        fc: float, epsilon: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
  ...


def gas_attenuation(fc, epsilon):
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
  return zenith_attenuation(fc=fc) / np.sin(epsilon)


def zenith_attenuation(fc: float) -> float:
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
def cal_deterministic_loss(distance: float, freq: float, epsilon: float) -> float:
  """Calculate the deterministic part of the loss.

  Args:
      distance (float): The distance between sat and ue.
      freq (float): The center freq.
      epsilon (float): The elevation angle pointing from ue to sat

  Returns:
      (float): The total deterministic loss (dB)
  """
  fspl = free_space(distance, freq)
  scpl = scintillation_loss(epsilon)
  gpl = gas_attenuation(freq, epsilon)
  return fspl + scpl + gpl


def cal_stochastic_loss(epsilon: float) -> float:
  """Calculate the stochastic part of the loss.

  Args:
      epsilon (float): The elevation angle pointing from ue to sat

  Returns:
      (float): The total stochastic loss (dB)
  """
  sf = shadow_fading(epsilon)
  return sf


def cal_total_loss(distance: float, freq: float, epsilon: float) -> float:
  """Calculate the total loss for sat and ue.

  Args:
      distance (float): The distance between sat and ue.
      freq (float): The center freq.
      epsilon (float): The elevation angle pointing from ue to sat

  Returns:
      (float): The total loss (dB)
  """
  det_loss = cal_deterministic_loss(distance=round(distance,
                                                   constant.CACHED_PRECISION),
                                    freq=round(freq,
                                               constant.CACHED_PRECISION),
                                    epsilon=round(epsilon,
                                                  constant.CACHED_PRECISION))
  stoch_loss = cal_stochastic_loss(epsilon=epsilon)

  return det_loss + stoch_loss
