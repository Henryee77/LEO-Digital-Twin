"""The beam module."""

from __future__ import annotations

from typing import Set

from ..util import constant
from ..util import Position
from ..util import util
from ..ground_user import User


class Beam(object):
  """The Beam class.

  Attribute:
    served_ue (Set[str]): The set of ues that this beam is serving
  """
  served_ue: Set[str]  # set of served users

  def __init__(self,
               index: int,
               center_point: Position,
               tx_power: float = constant.MIN_NEG_FLOAT,
               central_frequency: float = constant.DEFAULT_CENTRAL_FREQUENCY,
               bandwidth: float = constant.DEFAULT_BANDWIDTH,
               beamwidth_3db: float = constant.DEFAULT_BEAMWIDTH_3DB):
    """The __init__ funciton for beam.

    Args:
        center_point (Position) : The position of the beam center.
        tx_power (float): The transmitting power of the object (dBm).
        central_frequency (float) : The central frequency of the object.
        bandwidth (float) : The bandwidth of the object.
        beamwidth_3db (float) : The 3dB beamwidth (radian) of the object.

    Raise:
        ValueError: The value of the power, frequency and angle must be
                    non-negative.
    """
    self.__index = index
    self.tx_power = tx_power
    self.central_frequency = central_frequency
    self.bandwidth = bandwidth
    self.beamwidth_3db = beamwidth_3db
    self.center_point = center_point
    self.served_ue = set()

  def __str__(self) -> str:
    return (f'tx_power: {self.tx_power} dBm, '
            f'central_frequency: {self.central_frequency:.1e} Hz, '
            f'bandwidth: {self.bandwidth:.1e} Hz, '
            f'beamwidth_3db: {self.beamwidth_3db / constant.PI_IN_RAD} deg')

  @property
  def index(self):
    return self.__index

  @property
  def tx_power(self):
    return self._tx_power

  @tx_power.setter
  def tx_power(self, power: float):
    self._tx_power = power

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
  def bandwidth(self):
    return self._bandwidth

  @bandwidth.setter
  def bandwidth(self, bw: float):
    if bw > 0:
      self._bandwidth = bw
    else:
      raise ValueError('The value of the beandwidth must be positive.')

  @property
  def beamwidth_3db(self):
    return self._beamwidth_3db

  @beamwidth_3db.setter
  def beamwidth_3db(self, beamwidth: float):
    if beamwidth > 0:
      self._beamwidth_3db = beamwidth
    else:
      raise ValueError('The value of the 3dB beamwidth must be positive.')

  @property
  def center_point(self):
    return self._center_point

  @center_point.setter
  def center_point(self, center: Position):
    if center is None:
      raise ValueError('center cannot be None')
    self._center_point = center

  def add_serving(self, ue_name: str):
    """Adding the serving to ue.

    Args:
        ue_name (str): The name of the UE.
    """
    self.served_ue.add(ue_name)

  def remove_serving(self, ue_name: str):
    """Removing the serving to ue.

    Args:
        ue_name (str): The name of the UE.
    """
    if ue_name in self.served_ue:
      self.served_ue.remove(ue_name)
    else:
      raise ValueError(
          'Cannot remove the user which is not in the served dict')

    if not self.served_ue:
      self.tx_power = constant.MIN_NEG_FLOAT

  def calc_sinr(self,
                ue: User,
                tx_gain: float,
                channel_loss: float,
                interference_power: float,
                mode: str = 'run') -> float:
    """Calclate the SINR of user ue

    Args:
        ue (User): The user this beam is serving
        tx_gain (float): The tx antenna gain (in dB) this user gets
        interference_power (float): The interference power (in dBm)
                                    this user gets
        mode (str): the mode this function is running
                    (run or debug)

    Returns:
        float: The SINR (in dB)
    """
    noise_power = constant.THERMAL_NOISE_POWER + util.todb(self.bandwidth)
    n_and_i = util.todb(
        util.tolinear(interference_power) + util.tolinear(noise_power))

    sinr = self.tx_power + tx_gain + ue.rx_gain - channel_loss - n_and_i
    if mode == 'debug':
      print(f'Tx Power: {self.tx_power} dBm, '
            f'Tx Gain: {tx_gain}, '
            f'UE Rx Gain: {ue.rx_gain}, '
            f'Channel Loss: {channel_loss} dB, '
            f'Interference Power: {interference_power} dBm, '
            f'Noise power: {noise_power} dBm,'
            f'Interference and Noise: {n_and_i} dBm,'
            f'SINR: {sinr}')
    return sinr

  def has_interference(self, other_beam: Beam) -> bool:
    """Decide if there is interference between two beams.

    Args:
        other_beam (Beam): The another beam

    Returns:
        bool: The bool value for having interference or not
    """
    self_max_freq = self.central_frequency + 0.5 * self.bandwidth
    self_min_freq = self.central_frequency - 0.5 * self.bandwidth
    other_max_freq = other_beam.central_frequency + 0.5 * other_beam.bandwidth
    other_min_freq = other_beam.central_frequency - 0.5 * other_beam.bandwidth

    both_beam_is_on = self.tx_power > constant.MIN_NEG_FLOAT and other_beam.tx_power > constant.MIN_NEG_FLOAT
    self_freq_greater_and_overlap = (
        self_max_freq > other_min_freq and
        self.central_frequency <= other_beam.central_frequency)
    self_freq_smaller_and_overlap = (
        self_min_freq < other_max_freq and
        self.central_frequency >= other_beam.central_frequency)

    return (both_beam_is_on and self_freq_greater_and_overlap or
            self_freq_smaller_and_overlap)
