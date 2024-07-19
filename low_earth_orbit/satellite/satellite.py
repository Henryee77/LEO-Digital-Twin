"""satellite.py"""

from typing import List, Dict, Set
from typing import Tuple
import collections

from ..antenna import Antenna
from ..cell import CellTopology
from ..channel import channel
from ..util import Position
from ..util import constant
from ..ground_user import User


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
               max_power: float = constant.MAX_POWER,
               beam_alg: int = constant.DEFAULT_BEAM_SWEEPING_ALG):

    self._shell_index = shell_index
    self._plane_index = plane_index
    self._sat_index = sat_index
    self.position = position
    self.angle_speed = angle_speed
    self.cell_topo = cell_topo
    self.antenna = antenna
    self.max_power = max_power
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
  def training_latency(self) -> float:
    return ((self.cell_topo.training_beam_num + 1) * constant.T_BEAM
            + constant.T_FB + constant.T_ACK)

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

  def cal_rsrp(self, ue: User):
    """Calculate the rsrp with one ue.

    Args:
        ue (User): The target ue that in in servable range
    """
    for cell_index in self.cell_topo.training_beam:
      rsrp = self.cal_rsrp_one_beam(
          self.cell_topo.beam_list[cell_index].center_point, ue)
      ue.servable_add(self.name, cell_index, rsrp)

  def cal_rsrp_one_beam(self, beam_pos: Position, ue: User) -> float:
    """Calculate the rsrp with one beam.

    Args:
        beam_pos (Position): The beam center position
        ue (User): The target ue that in in servable range

    Returns:
        (float): The rx power in dBW
    """
    epsilon = self.position.cal_elevation_angle(beam_pos)
    dis_sat_ue = self.position.calculate_distance(ue.position)
    theta = self.position.angle_between_targets(beam_pos, ue.position)

    antenna_gain = float(self.antenna.calc_antenna_gain(theta))

    path_loss = channel.cal_total_loss(distance=dis_sat_ue,
                                       freq=self.antenna.central_frequency,
                                       epsilon=epsilon)

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

      path_loss = channel.cal_total_loss(distance=dis_sat_ue,
                                         freq=self.antenna.central_frequency,
                                         epsilon=epsilon)
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

  def scan_beam(self):
    """Calculate the RSRP of the training beam set"""
    for ue in self.servable:
      self.cal_rsrp(ue)

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

  def select_train_by_topo(self, ues: List[User]):
    """Select the training beams

    Args:
        ues (List[User]): The users this satellite is serving
    """
    self.servable = list(filter(self.filter_ue, ues))
    self.cell_topo.training_beam.clear()
    for ue in self.servable:
      self.cell_topo.training_beam = self.cell_topo.training_beam.union(
          self.cell_topo.find_nearby(ue.position))

    if self.cell_topo.training_beam:
      self.scan_beam()

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
