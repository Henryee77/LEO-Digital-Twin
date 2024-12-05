"""satellite.py"""

from typing import List, Dict, Set
from typing import Tuple
import collections
import math
import numpy as np

from ..antenna import Antenna
from ..cell import CellTopology, Beam
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
    self.antenna_list = [Antenna() for _ in range(self.cell_topo.cell_number)]
    if channel:
      self.wireless_channel = channel
    else:
      self.wireless_channel = Channel()

    self.__max_power = max_power
    self.__min_power = min_power
    self.total_bandwidth = total_bandwidth
    self.beam_alg = beam_alg
    self.__bs_latency = 0
    self.large_param_variation = 0
    self.mid_param_variation = 0
    self.small_param_variation = 0
    self.nakagami_m_var = 0
    self.los_power_var = 0
    self.rx_power_var = 0

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
  def nakagami_m_var(self):
    return self.__nakagami_m_var

  @nakagami_m_var.setter
  def nakagami_m_var(self, var):
    self.__nakagami_m_var = np.clip(var, -constant.MAX_L_VARIATION_PERCENT, constant.MAX_L_VARIATION_PERCENT)

  @property
  def los_power_var(self):
    return self.__los_power_var

  @los_power_var.setter
  def los_power_var(self, var):
    self.__los_power_var = np.clip(var, -constant.MAX_M_VARIATION_PERCENT, constant.MAX_M_VARIATION_PERCENT)

  @property
  def rx_power_var(self):
    return self.__rx_power_var

  @rx_power_var.setter
  def rx_power_var(self, var):
    self.__rx_power_var = np.clip(var, -constant.MAX_S_VARIATION_PERCENT, constant.MAX_S_VARIATION_PERCENT)

  @property
  def intrinsic_beam_sweeping_latency(self) -> float:
    return self.cell_topo.training_beam_num * constant.T_BEAM

  @property
  def beam_sweeping_latency(self) -> float:
    return self.__bs_latency

  @beam_sweeping_latency.setter
  def beam_sweeping_latency(self, bs_latency):
    self.__bs_latency = bs_latency

  @property
  def ues_feedback_latency(self) -> float:
    return constant.T_FB * len(self.serving_ues)

  @property
  def ack_latency(self) -> float:
    return constant.T_ACK * len(self.serving_ues)

  @property
  def avg_ue_prop_latency(self) -> float:
    distance_to_ues = [self.position.calculate_distance(ue.position) for ue in self.serving_ues]
    if len(distance_to_ues) == 0:
      return 0
    return sum(distance_to_ues) / len(distance_to_ues) / constant.LIGHT_SPEED

  @property
  def beam_training_latency(self) -> float:
    return self.beam_sweeping_latency + self.ues_feedback_latency + self.ack_latency + 2 * self.avg_ue_prop_latency

  def trans_latency(self, data_size: int, target: User) -> float:
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

  def reset_beamwidth(self):
    for antenna in self.antenna_list:
      antenna.beamwidth_3db = constant.DEFAULT_BEAMWIDTH_3DB
    for i in range(self.cell_topo.cell_number):
      self.cell_topo.set_beamwidth(i, constant.DEFAULT_BEAMWIDTH_3DB)

  def reset_channel_params(self):
    self.nakagami_m_var = 0
    self.los_power_var = 0
    self.rx_power_var = 0

  def update_channel_params(self):
    l_high = constant.MAX_L_VARIATION_PERCENT * 0.1
    l_low = -l_high
    m_high = constant.MAX_M_VARIATION_PERCENT * 0.1
    m_low = -m_high
    s_high = constant.MAX_S_VARIATION_PERCENT * 0.1
    s_low = -s_high
    self.nakagami_m_var = self.nakagami_m_var + np.random.uniform(l_low, l_high)
    self.los_power_var = self.los_power_var + np.random.uniform(m_low, m_high)
    self.rx_power_var = self.rx_power_var + np.random.uniform(s_low, s_high)

  def update_pos(self, time: float):
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

    power_dict = self.export_power_dict()
    self.clear_power()
    for beam_index in training_beams:
      self.set_beam_power(beam_index, self.max_power)
      rsrp = self.sinr_of_user(ue, beam_index)
      self.set_beam_power(beam_index, constant.MIN_NEG_FLOAT)

      if save_servable:
        ue.servable_add(self.name, beam_index, rsrp)
      rsrp_list[beam_index] = rsrp

    self.import_power_dict(power_dict)
    return rsrp_list

  def sinr_of_user(self,
                   ue: User,
                   serving_beam_index: int = None,
                   i_power: float = 0,
                   mode: str = 'run') -> float:
    """Get the SINR of the given users.

    Args:
        ue (User): The user this satellite is serving.
        serving_beam (int): The target beam to calculate SINR.
        i_power (float): The total interference power ue gets.
        mode (str): The mode this function is running.
                    (run or debug)

    Returns:
        float: The SINR of the ue under the serving beam.
    """

    if serving_beam_index is None:
      serving_beam = self.cell_topo.serving_beam_of_ue(ue)
    else:
      serving_beam = self.cell_topo.beam_list[serving_beam_index]

    beam_pos = serving_beam.center_point
    beam_index = serving_beam.index
    epsilon = self.position.cal_elevation_angle(beam_pos)
    dis_sat_ue = self.position.calculate_distance(ue.position)
    theta = self.position.angle_between_targets(beam_pos, ue.position)

    tx_gain = float(self.antenna_list[beam_index].calc_antenna_gain(theta))
    path_loss = self.wireless_channel.cal_total_loss(distance=dis_sat_ue,
                                                     freq=self.antenna_list[beam_index].central_frequency,
                                                     elevation_angle=epsilon,
                                                     nakagami_m=constant.NAKAGAMI_PARAMETER *
                                                     (1 + self.nakagami_m_var),
                                                     rx_power_ratio=constant.TOTAL_POWER_RECEIVED *
                                                     (1 + self.rx_power_var),
                                                     los_power_ratio=constant.LOS_COMPONENT_POWER *
                                                     (1 + self.los_power_var),
                                                     water_vapor_density=constant.GROUND_WATER_VAP_DENSITY *
                                                     (1 + ue.water_vap_var),
                                                     temperature=constant.GROUND_TEMPERATURE *
                                                     (1 + ue.temperature_var),
                                                     atmos_pressure=constant.GROUND_ATMOS_PRESSURE *
                                                     (1 + ue.atmos_press_var))
    if 'debug':
      print(f'{(1 + self.nakagami_m_var)},'
            f'{(1 + self.rx_power_var)},'
            f'{(1 + self.los_power_var)},'
            f'{(1 + ue.water_vap_var)},'
            f'{(1 + ue.temperature_var)},'
            f'{(1 + ue.atmos_press_var)}')

    channel_loss = path_loss

    return serving_beam.calc_sinr(
        ue=ue,
        tx_gain=tx_gain,
        channel_loss=channel_loss,
        interference_power=i_power,
        mode=mode,
    )

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
    self.antenna_list[beam_idx].beamwidth_3db = beamwidth
    self.cell_topo.set_beamwidth(beam_idx, beamwidth)

  def select_train_by_topo(self, ues: List[User]) -> Dict[str, List[float]]:
    """Select the training beams

    Args:
        ues (List[User]): The users this satellite is serving
    """

    self.servable = list(filter(self.filter_ue, ues))
    self.cell_topo.clear_training_beam()

    for ue in self.servable:
      new_training_beam = self.cell_topo.hobs(ue)
      self.cell_topo.add_training_beam(new_training_beam)

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
    return self.all_power > self.max_power * 1.0000001  # dealing with the precision in float calculation

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

  def export_power_dict(self) -> Dict[int, float]:
    """Export the tx power of every beam into a dictionary.

    Returns:
        Dict[int, float]: {beam_index: tx_power}
    """
    return self.cell_topo.export_power_dict()

  def import_power_dict(self, power_dict: Dict[int, float]):
    """Import the tx power data from a dictionary.

    Args:
        power_dict (Dict[int, float]): {beam_index: tx_power}
    """
    self.cell_topo.import_power_dict(power_dict)
