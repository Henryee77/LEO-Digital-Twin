"""The antenna module."""

from typing import Union

import numpy as np
import numpy.typing as npt
from scipy.special import jv

from ..util import constant
from ..util import util


class Antenna(object):
  """The Antenna class."""

  def __init__(self,
               aperture: float = constant.DEFAULT_APERTURE_SIZE,
               central_frequency: float = constant.DEFAULT_CENTRAL_FREQUENCY,
               beamwidth_3db: float = constant.DEFAULT_BEAMWIDTH_3DB,
               efficiency: float = constant.DEFAULT_ANTENNA_EFFICIENCY):
    """The __init__ funciton for antenna.

    Args:
        aperture (float): The aperture size of the object.
        central_frequency (float) : The central frequency of the object.
        beamwidth_3db (float) : The 3dB beamwidth of the object.
        efficiency (float) : The antenna efficiency of the object.

    Raise:
        ValueError: The value of the aperture, frequency and angle must be
                    non-negative.
    """
    self.aperture = aperture
    self.central_frequency = central_frequency
    self.beamwidth_3db = beamwidth_3db
    self._efficiency = efficiency

  @property
  def maximum_gain(self):
    return self.efficiency * (4 * constant.PI * self.aperture) / (
        constant.LIGHT_SPEED / self.central_frequency)**2

  @property
  def aperture(self):
    return self._aperture

  @aperture.setter
  def aperture(self, aperture: float):
    if aperture > 0:
      self._aperture = aperture
    else:
      raise ValueError(
          'The value of the central aperture size must be positive.')

  @property
  def central_frequency(self):
    return self._central_frequency

  @central_frequency.setter
  def central_frequency(self, fc: float):
    if fc > 0:
      self._central_frequency = fc
    else:
      raise ValueError('The value of the central frequency must be positive.')

  @property
  def beamwidth_3db(self):
    return self.__beamwidth_3db

  @beamwidth_3db.setter
  def beamwidth_3db(self, beamwidth_3db: float):
    if beamwidth_3db > 0:
      self.__beamwidth_3db = beamwidth_3db
    else:
      raise ValueError('The value of the angle must be positive.')

  @property
  def efficiency(self):
    return self._efficiency

  def calc_antenna_gain(
          self, theta: Union[float, npt.NDArray]) -> Union[float, npt.NDArray]:
    """Calculate the antenna gain based on the off-boresight angle (radian)

    Returns:
        The antenna gain (dB) of different off-boresight angle
    """
    mu = constant.ANT_GAIN_COEFF * np.sin(theta) / np.sin(self.beamwidth_3db)
    if isinstance(mu, np.ndarray):
      mu[mu == 0] = constant.MIN_POSITIVE_FLOAT  # avoid divide by zero
    else:
      if mu == 0:
        mu = constant.MIN_POSITIVE_FLOAT
    normalized_gain = (jv(1, mu) / (2 * mu) + 36 * jv(3, mu) / (mu**3))**2

    return util.todb(self.maximum_gain * normalized_gain)
