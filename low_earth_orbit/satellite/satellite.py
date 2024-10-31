"""satellite.py"""

from typing import List, Dict, Set
from typing import Tuple
import collections
import math

from ..antenna import Antenna
from ..cell import CellTopology
from ..channel import Channel
from ..util import Position
from ..util import constant
from ..ground_user import User
from ..util import util


class Satellite(object):
  """The satellite class."""
  servable: List[User]

  def __init__(self,
               shell_index: int,
               plane_index: int,
               sat_index: int,
               position: Position,
               angle_speed: float,
               cell_topo: CellTopology,
               antenna: Antenna,
               channel: Channel,
               max_power: float = constant.MAX_POWER,
               min_power: float = constant.MIN_POWER,
               total_bandwidth=constant.DEFAULT_BANDWIDTH,
               beam_alg: int = constant.DEFAULT_BEAM_SWEEPING_ALG):

    self._shell_index = shell_index
    self._plane_index = plane_index
    self._sat_index = sat_index
    self.position = position
    self.angle_speed = angle_speed
    self.cell_topo = cell_topo
    self.antenna = antenna
    self.wireless_channel = channel
    self.__max_power = max_power
    self.__min_power = min_power
    self.total_bandwidth = total_bandwidth
    self.beam_alg = beam_alg

  @property
  def shell_index(self):
    return self._shell_index

  @property
  def plane_index(self):
    return self._plane_index

  @property
  def sat_index(self):
    return self._sat_index

  @property
  def name(self):
    return f'{self.shell_index}_{self.plane_index}_{self.sat_index}'

  @property
  def position(self):
    return self._position

  @position.setter
  def position(self, pos: Position):
    if pos is not None:
      self._position = pos
    else:
      raise ValueError('Cannot set the position of satellite to None')

  @property
  def max_power(self):
    """Maximum power in dBm"""
    return self.__max_power

  @property
  def min_power(self):
    """Minimum power in dBm"""
    return self.__min_power

  @property
  def min_beamwidth(self) -> float:
    return constant.DEFAULT_MIN_BEAMWIDTH

  @property
  def max_beamwidth(self) -> float:
    return constant.DEFAULT_MAX_BEAMWIDTH

  @property
  def beam_number(self) -> int:
    """The number of beams

    Returns:
        int: beam number
    """
    return self.cell_topo.cell_number

  @property
  def all_power(self) -> float:
    """The total tx power of the satellite

    Returns:
        float: total tx poewr in dBm
    """
    return self.cell_topo.all_beam_power()

  @property
  def serving_ues(self) -> List[User]:
    return [ue for ue in self.servable if ue.name in self.cell_topo.serving.keys()]

  @property
  def beam_sweeping_latency(self) -> float:
    return self.cell_topo.training_beam_num * constant.T_BEAM

  @property
  def ues_feedback_latency(self) -> float:
    return constant.T_FB * len(self.serving_ues)

  @property
  def ack_latency(self) -> float:
    return constant.T_ACK * len(self.serving_ues)

  @property
  def avg_ue_prop_latency(self) -> float:
    distance_to_ues = [self.position.calculate_distance(ue.position) for ue in self.serving_ues]
    return sum(distance_to_ues) / len(distance_to_ues) / constant.LIGHT_SPEED

  @ property
  def beam_training_latency(self) -> float:
    return self.beam_sweeping_latency + self.ues_feedback_latency + self.ack_latency + 2 * self.avg_ue_prop_latency

  def trans_latency(self, data_size: int, target) -> float:
    """Transmission latency

    Args:
        data_size (int): byte
        target (User): transmission target

    Returns:
        float: latency
    """
    max_rsrp = max(self.cal_rsrp(ue=target,
                                 training_beams=set([i for i in range(self.cell_topo.cell_number)]),
                                 save_servable=False))
    noise_power = constant.THERMAL_NOISE_POWER + util.todb(self.total_bandwidth)
    return data_size / (self.total_bandwidth * math.log2(1 + util.tolinear(max_rsrp - noise_power)))

  def clear_power(self):
    """Set all the beam power to zero"""
    self.cell_topo.clear_power()

  def update_pos(self, time: float = constant.TIMESLOT):
    """Update the position by the given time

    Args:
        time (float): The elapsed time. Defaults to constant.TIMESLOT.
    """
    updated_orbital = self.position.orbital
    updated_orbital.small_omega += self.angle_speed * time
    updated_orbital.large_omega -= constant.ANGULAR_SPEED_EARTH * time
    self.position.orbital = updated_orbital
    self.cell_topo.center_point = self.position

  def filter_ue(self, ue: User) -> bool:
    distance = self.position.calculate_distance(ue.position)
    epsilon = self.position.cal_elevation_angle(ue.position)
    epsilon_bool = epsilon > constant.EPSILON_SERVING_THRESHOLD
    distance_bool = distance < 2 * (self.position.geodetic.height -
                                    constant.R_EARTH)
    return epsilon_bool and distance_bool

  def cal_rsrp(self, ue: User, training_beams: Set[int] = None, save_servable=True) -> List[float]:
    """Calculate the rsrp with one ue.

    Args:
        ue (User): The target ue that in in servable range

    Returns:
        (List[float]): The List of rsrp for each beam.
    """
    rsrp_list = [constant.MIN_NEG_FLOAT] * self.cell_topo.cell_number
    if training_beams is None:
      training_beams = self.cell_topo.training_beam

    for cell_index in training_beams:
      rsrp = self.cal_rsrp_one_beam(
          self.cell_topo.beam_list[cell_index].center_point, ue)
      if save_servable:
        ue.servable_add(self.name, cell_index, rsrp)
      rsrp_list[cell_index] = rsrp
    return rsrp_list

  def cal_rsrp_one_beam(self, beam_pos: Position, ue: User) -> float:
    """Calculate the rsrp with one beam.

    Args:
        beam_pos (Position): The beam center position
        ue (User): The target ue that in in servable range

    Returns:
        (float): The rx power in dBm
    """
    epsilon = self.position.cal_elevation_angle(beam_pos)
    dis_sat_ue = self.position.calculate_distance(ue.position)
    theta = self.position.angle_between_targets(beam_pos, ue.position)

    antenna_gain = float(self.antenna.calc_antenna_gain(theta))

    path_loss = self.wireless_channel.cal_total_loss(distance=dis_sat_ue,
                                                     freq=self.antenna.central_frequency,
                                                     elevation_angle=epsilon)

    rx_power = constant.MAX_POWER - path_loss + antenna_gain + ue.rx_gain

    return rx_power

  def sinr_of_users(self,
                    serving_ue: List[User],
                    i_power: List[float],
                    mode: str = 'run') -> List[float]:
    """Get the SINR of its serving users

    Args:
        serving_ue (list[User]): The users this satellite is serving
        i_power (float): The total interference power each ue gets
        mode (str): the mode this function is running
                    (run or debug)

    Returns:
        list[float]: The SINR of each ue
    """
    tx_gain, channel_loss = [], []
    for ue in serving_ue:
      beam_pos = self.cell_topo.beam_pos_of_serving_ue(ue)
      epsilon = self.position.cal_elevation_angle(beam_pos)
      dis_sat_ue = self.position.calculate_distance(ue.position)
      theta = self.position.angle_between_targets(beam_pos, ue.position)

      tx_gain.append(float(self.antenna.calc_antenna_gain(theta)))

      path_loss = self.wireless_channel.cal_total_loss(distance=dis_sat_ue,
                                                       freq=self.antenna.central_frequency,
                                                       elevation_angle=epsilon)
      channel_loss.append(path_loss)

    return self.cell_topo.sinr_of_users(serving_ue=serving_ue,
                                        tx_gain=tx_gain,
                                        channel_loss=channel_loss,
                                        i_power=i_power,
                                        mode=mode)

  def add_cell_topo_info(self, ue_name: str, beam_idx: int):
    """Update the information of which beam is serving the ue"""
    self.cell_topo.add_serving(ue_name, beam_idx)

  def drop_cell_topo_info(self, ue_name: str):
    """Update the information of which beam is serving the ue"""
    self.cell_topo.remove_serving(ue_name)

  def scan_beams(self) -> Dict[str, List[float]]:
    """Calculate the RSRP of the training beam set"""
    ues_sinr = {}
    for ue in self.servable:
      ues_sinr[ue.name] = self.cal_rsrp(ue)

    return ues_sinr

  def set_beam_power(self, beam_idx: int, tx_power: float):
    """Set the tx power of the beam

    Args:
        beam_idx (int): beam index
        tx_power (float): tx power of the beam

    Raises:
        ValueError: Exceeds the max power of the satellite
    """
    self.cell_topo.set_beam_power(beam_idx, tx_power)
    if self.exceed_max_power():
      self.cell_topo.print_all_beams()
      raise ValueError('Exceeds the max power of the satellite')

  def set_beamwidth(self, beam_idx: int, beamwidth: float):
    """Set the 3dB beamwidth of the beam

    Args:
        beam_idx (int): beam index
        beamwidth (float): 3dB beamwidth
    """
    self.cell_topo.set_beamwidth(beam_idx, beamwidth)

  def select_train_by_topo(self, ues: List[User]) -> Dict[str, List[float]]:
    """Select the training beams

    Args:
        ues (List[User]): The users this satellite is serving
    """
    self.servable = list(filter(self.filter_ue, ues))
    self.cell_topo.clear_training_beam()
    for ue in self.servable:
      new_training_beam = self.cell_topo.find_nearby(ue)
      self.cell_topo.add_training_beam(new_training_beam)

    if self.cell_topo.training_beam:
      ues_sinr = self.scan_beams()
    else:
      ues_sinr = {}
      for ue in ues:
        ues_sinr[ue.name] = [constant.MIN_NEG_FLOAT] * self.cell_topo.cell_number

    return ues_sinr

  def get_ue_data(
          self, ues: List[User]) -> Dict[str, collections.deque[Tuple[str, int]]]:
    """Get the serving history of the users if the size of its history data
       is greater than the training window size
    Args:
        ues (List[User]): The users

    Returns:
        Dict[str, collections.deque[Tuple[str, int]]]: The history data of the users
    """
    self.servable = list(filter(self.filter_ue, ues))

    return {user.name: user.serving_history for user in self.servable}

  def exceed_max_power(self) -> bool:
    """Check if the total beam power exceeds the max power of the satellite

    Returns:
        bool: True if exceeds
    """
    return self.all_power > self.max_power

  def assign_train_set(self, train_set: Set[int]):
    """Assign the training set to the satellite

    Args:
        train_set (Set[int]): The training set of the satellite
    """
    self.cell_topo.training_beam = train_set

  def get_beam_info(self, beam_idx: int) -> Tuple[float, float, float, int]:
    """Get the info of the beam

    Args:
        beam_idx (int): The indx of the beam

    Returns:
        Tuple[float, float, float, int]: tx_power, central_frequency, bandwidth, served ue number
    """
    return self.cell_topo.get_beam_info(beam_idx)
