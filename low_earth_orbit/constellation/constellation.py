"""constellation.py"""

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Set
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from ..util import constant
from ..util import util
from ..util import Orbital
from ..util import Position
from ..util import SatBeamID
from ..satellite import Satellite
from ..cell import CellTopology
from ..antenna import Antenna
from ..ground_user import User
from ..channel import Channel


@dataclass
class ConstellationData:
  """The basic data for Constellation."""
  plane_num: int
  sat_per_plane: int
  sat_height: int
  inclination_angle: int


class Constellation(object):
  """All Constellation.

  Attributes:
      all_sat: The dict that maps the name of the satellite to satellite object
  """
  all_sat: Dict[str, Satellite]

  def __init__(
      self,
      setup_data_list: List[ConstellationData],
      args=None,
  ):
    """The constructer of Constellation Class.

    Args:
        setup_data_list (list[ConstellationData]): All data for constellation.
        args: args
    """
    self.args = args
    self.shell_num = len(setup_data_list)
    self.setup_data_list = setup_data_list
    self.sat_number = sum(
        data.plane_num * data.sat_per_plane for data in self.setup_data_list)
    self.angular_speed_orbital = [
        util.calc_sat_angular_speed(data.sat_height + constant.R_EARTH)
        for data in self.setup_data_list
    ]
    self.all_sat = {}
    self.create_all_sat()

  def create_all_sat(self):
    """Creates the all sat for one shell."""
    for shell_index, item in enumerate(self.setup_data_list):
      for plane_index in range(item.plane_num):
        for sat_index in range(item.sat_per_plane):
          self.create_sat(item, shell_index, plane_index, sat_index)

  def create_sat(
      self,
      setup_data: ConstellationData,
      shell_index: int,
      plane_index: int,
      sat_index: int,
  ):
    """Creates the sat.

    Args:
        setup_data (ConstellationData): The data for constellation.
        shell_index (int): The shell index in the constellation.
        plane_index (int): The plane index in the shell.
        sat_index (int): The sat index in the plane.
    """
    c_0 = 2 * constant.PI / setup_data.sat_per_plane
    c_1 = 2 * constant.PI / (setup_data.plane_num * setup_data.sat_per_plane)
    c_2 = 2 * constant.PI / (self.shell_num * setup_data.sat_per_plane)
    small_o = sat_index * c_0 + plane_index * c_1 + shell_index * c_2
    large_o = plane_index * 2 * constant.PI / setup_data.plane_num
    large_o += constant.ORIGIN_LONG * constant.PI_IN_RAD

    cal_temp = constant.ORIGIN_LATI * math.cos(setup_data.inclination_angle)
    cal_temp *= constant.PI_IN_RAD

    if shell_index % 2:
      large_o += (cal_temp + constant.PI)
    else:
      large_o -= cal_temp

    satellite_pos = Orbital(inclination=setup_data.inclination_angle,
                            small_omega=small_o,
                            large_omega=large_o,
                            radius=setup_data.sat_height + constant.R_EARTH)

    projection_point = satellite_pos.to_geodetic()
    projection_point.height = constant.R_EARTH

    sat_obj = Satellite(shell_index=shell_index,
                        plane_index=plane_index,
                        sat_index=sat_index,
                        angle_speed=self.angular_speed_orbital[shell_index],
                        position=Position(orbital=satellite_pos),
                        cell_topo=CellTopology(center_point=Position(geodetic=projection_point),
                                               cell_layer=self.args.cell_layer_num)
                        )

    self.all_sat[sat_obj.name] = sat_obj

  def update_sat_position(self, time=constant.MOVING_TIMESLOT):
    """Update the position of all satellite after one timeslot."""
    for sat in self.all_sat.values():
      sat.update_pos(time=time)

  def plot_constellation(self,
                         time: float,
                         shell_color: List[str],
                         fig: Optional[plt.Figure] = None,
                         ax: Optional[plt.Axes] = None):
    """Plot the constellation"""
    if fig is None or ax is None:
      fig = plt.figure(figsize=(16, 9), dpi=80)
      ax = plt.axes(projection='3d')
      ax.set_aspect('auto')
    ax.clear()

    u, v = np.mgrid[0:2 * constant.PI:20j, 0:constant.PI:10j]
    earth_x = constant.R_EARTH * np.cos(u) * np.sin(v)
    earth_y = constant.R_EARTH * np.sin(u) * np.sin(v)
    earth_z = constant.R_EARTH * np.cos(v)
    ax.plot_wireframe(earth_x, earth_y, earth_z, color='blue')

    for sat in self.all_sat.values():
      sat_x, sat_y, sat_z = sat.position.cartesian.get()
      ax.scatter(sat_x,
                 sat_y,
                 sat_z,
                 s=constant.SAT_MARKER_SIZE,
                 c=shell_color[sat.shell_index])

    plt.title(f'time = {time}',
              fontdict={
                  'fontsize': 24,
                  'fontweight': 'medium'
    })
    plt.pause(0.1)

  def scan_ues(self, ues: List[User], sat_name_list: List[str] = None, scan_mode=None) -> Dict[str, Dict[str, List[float]]]:
    """Select the training beam and calculate the RSRP,
       and add the servable data to the user

    Args:
        ues (List[User]): The users
    """
    for ue in ues:
      ue.servable_clear()

    if sat_name_list is None:
      sat_name_list = self.all_sat.keys()

    # Obtain beam training set
    sat_ues_sinr = {}
    temp_power_dict = {}
    for sat_name in sat_name_list:
      temp_power_dict[sat_name] = self.all_sat[sat_name].export_power_dict()
      self.all_sat[sat_name].select_train_by_topo(ues)

    # Beam sweeping schemes
    if scan_mode == 'SCBS':
      sat_ues_sinr = self.SCBS(ues, sat_name_list)
    elif scan_mode == 'SSBS':
      sat_ues_sinr = self.SSBS(ues, sat_name_list)
    elif scan_mode == 'ABS':
      sat_ues_sinr = self.ABS(ues, sat_name_list)
    else:
      raise ValueError(f'No \'{scan_mode}\' beam sweeping mode.')

    for sat_name in sat_name_list:
      self.all_sat[sat_name].import_power_dict(temp_power_dict[sat_name])

    return sat_ues_sinr

  def SCBS(self, ues: List[User], sat_name_list: List[str]) -> Dict[str, Dict[str, List[float]]]:
    total_sweeping_time = 0
    sat_ues_sinr = {}
    # Beam sweeping
    for sat_name in sat_name_list:
      sat = self.all_sat[sat_name]
      if sat.cell_topo.training_beam:
        ues_sinr = sat.scan_beams()
        total_sweeping_time += sat.intrinsic_beam_sweeping_latency
      else:
        ues_sinr = self._no_training_result(ues=ues, sat=sat)
      sat_ues_sinr[sat_name] = ues_sinr

    # Calculate beam sweeping latency
    for sat_name in sat_name_list:
      self.all_sat[sat_name].beam_sweeping_latency = total_sweeping_time

    return sat_ues_sinr

  def SSBS(self, ues: List[User], sat_name_list: List[str]) -> Dict[str, Dict[str, List[float]]]:
    # Beam sweeping
    total_sweeping_time = 0
    training_beams = {}
    max_training_num = 0
    for sat_name in sat_name_list:
      sat = self.all_sat[sat_name]
      training_beams[sat_name] = list(sat.cell_topo.training_beam)
      random.shuffle(training_beams[sat_name])
      max_training_num = max(max_training_num, len(training_beams[sat_name]))
      total_sweeping_time = max(sat.intrinsic_beam_sweeping_latency, total_sweeping_time)

    sat_ues_sinr = self._cal_SSBS_sinr(ues=ues,
                                       training_beams=training_beams,
                                       max_training_num=max_training_num)

    # Calculate beam sweeping latency
    for sat_name in sat_name_list:
      self.all_sat[sat_name].beam_sweeping_latency = total_sweeping_time

    return sat_ues_sinr

  def ABS(self, ues: List[User], sat_name_list: List[str]) -> Dict[str, Dict[str, List[float]]]:
    sat_ues_sinr = {}

    # Beam sweeping
    for sat_name in sat_name_list:
      sat = self.all_sat[sat_name]
      ues_sinr = {}
      if sat.cell_topo.training_beam:
        for ue in sat.servable:
          ues_sinr[ue.name] = self._cal_ABS_sinr(ue=ue,
                                                 sat=sat,
                                                 online_beam_set=set(servable_ue.last_serving
                                                                     for servable_ue in sat.servable
                                                                     if (servable_ue.last_serving is not None
                                                                         and (ue.last_serving is None or servable_ue.last_serving[0] != ue.last_serving[0])))
                                                 )
      else:
        ues_sinr = self._no_training_result(ues=ues, sat=sat)
      sat_ues_sinr[sat_name] = ues_sinr

    # Calculate beam sweeping latency
    for sat_name in sat_name_list:
      self.all_sat[sat_name].beam_sweeping_latency = self.all_sat[sat_name].intrinsic_beam_sweeping_latency

    return sat_ues_sinr

  def _cal_SSBS_sinr(self,
                     ues: List[User],
                     training_beams: Dict[str, List[int]],
                     max_training_num: int
                     ) -> Dict[str, float]:
    sat_ues_sinr = {}
    for sat_name in training_beams:
      self.all_sat[sat_name].clear_power()
      sat_ues_sinr[sat_name] = self._no_training_result(ues, self.all_sat[sat_name])

    for i in range(max_training_num):
      cur_training_set = set((sat_name, training_beams[sat_name][i])
                             for sat_name in training_beams
                             if len(training_beams[sat_name]) > i)

      for satbeam in cur_training_set:
        for ue in ues:
          sat_name, beam_idx = satbeam
          sat = self.all_sat[sat_name]
          sat.set_beam_power(beam_idx, sat.max_power)

          interference_power = self.cal_transmission_interference(ue, cur_training_set, satbeam)
          training_sinr = self.all_sat[sat_name].sinr_of_user(ue=ue,
                                                              serving_beam_index=beam_idx,
                                                              i_power=interference_power)
          sat_ues_sinr[sat_name][ue.name][beam_idx] = training_sinr
          ue.servable_add(name_sat=sat_name,
                          beam_num=beam_idx,
                          rsrp=training_sinr)

          sat.set_beam_power(beam_idx, constant.MIN_NEG_FLOAT)

    return sat_ues_sinr

  def _cal_ABS_sinr(self,
                    ue: User,
                    sat: Satellite,
                    online_beam_set: set[SatBeamID],
                    ) -> List[float]:
    """Asynchronous Beam Sweeping

    Args:
        ue (User): ue
        sat (Satellite): training satellite
        online_beam_set (set[SatBeamID]): interference sources

    Returns:
        List[float]: training sinr of each training beam in sat
    """

    training_sinr = [constant.MIN_NEG_FLOAT] * sat.cell_topo.cell_number
    training_beam_set = sat.cell_topo.training_beam

    sat.clear_power()
    interference_power = self.cal_transmission_interference(ue, online_beam_set)

    for beam_idx in training_beam_set:
      sat.set_beam_power(beam_idx, sat.max_power)

      training_sinr[beam_idx] = sat.sinr_of_user(
          ue=ue,
          serving_beam_index=beam_idx,
          i_power=interference_power,
      )

      ue.servable_add(name_sat=sat.name,
                      beam_num=beam_idx,
                      rsrp=training_sinr[beam_idx])

      sat.set_beam_power(beam_idx, constant.MIN_NEG_FLOAT)

    return training_sinr

  def cal_transmission_sinr(self,
                            ues: List[User],
                            mode: str = 'run',
                            interference_beams: Set[SatBeamID] | None = None) -> Dict[str, float]:
    """Calculate the SINR of users.

    This function is only supposed to be used when calculating the SINR
    of data trasmission. 

    Args:
        ues (List[User]): users
        serving_satbeam_dict (Dict[str, SatBeamID] | None): Default is None if you don't want to specify the beam. If the input is None it will select the current serving beam of the ue.
        mode (str): The execution mode of this method
                    1. 'run'
                    2. 'debug' (will print detail info)
        interferece_beams (Set[SatBeamID] | None): custom defined interferece beams

    Returns:
        Dict[str, float]: SINR of each user
    """
    last_serving_set = set(ue.last_serving for ue in ues if ue.last_serving is not None)
    if interference_beams is not None:
      last_serving_set.update(interference_beams)

    sinr = {}
    for ue in ues:
      serving_satbeam = ue.last_serving

      if serving_satbeam is None:  # The ue has no service
        sinr[ue.name] = math.nan
      else:
        interference_power = self.cal_transmission_interference(ue, last_serving_set, serving_satbeam)
        best_sat = self.all_sat[serving_satbeam[0]]
        sinr[ue.name] = best_sat.sinr_of_user(
            ue=ue,
            serving_beam_index=serving_satbeam[1],
            i_power=interference_power,
            mode=mode,
        )

    return sinr

  def cal_transmission_interference(self, ue: User, online_beam_set: Set[SatBeamID], serving_satbeam: SatBeamID | None = None) -> float:
    """Calculate the interference during the transmission phase of the ue.

    Args:
        ue (User): ue
        online_beam_set (Set[SatBeamID]): Set of the beam index which the beam is turned on
        serving_satbeam (SatBeamID | None): serving beam

    Returns:
        float: interference power (dBm) the ue is suffering
    """
    i_power = constant.MIN_POSITIVE_FLOAT  # in linear
    # print(online_beam_set)
    for sat_beam in online_beam_set:
      if serving_satbeam == sat_beam:
        continue
      sat_name, beam_idx = sat_beam
      sat = self.all_sat[sat_name]
      i_power += util.tolinear(sat.sinr_of_user(ue=ue,
                                                serving_beam_index=beam_idx,
                                                )  # rsrp of the interference beam
                               )
    # print(util.todb(i_power), 'end')
    return util.todb(i_power)  # dBm

  def set_beam_serving(self, sat_beam: SatBeamID, tx_power: float,
                       served_ues: List[User]):
    """Set the serving information with power

    Args:
        sat_beam (SatBeamID): The index of the serving satellite and beam
        tx_power (float): The tx power
        served_ues (List[User]): The users served by this SatBeam

    Raises:
        ValueError: The name of the serving satellite doesn't exist
    """
    name_sat, beam_idx = sat_beam
    if name_sat not in self.all_sat:
      raise ValueError('Constellation doesn\'t have this satellite')
    self.all_sat[name_sat].set_beam_power(beam_idx, tx_power)
    for ue in served_ues:
      self.add_serving(sat_beam, ue.name)

  def set_beamwidth(self, sat_beam: SatBeamID, beamwidth: float):
    """Set the beamwidth to the specify SatBeam

    Args:
        sat_beam (SatBeamID): The index of the satellite and the beam
        beamwidth (float): The 3dB beamwidth

    Raises:
        ValueError: The name of the serving satellite doesn't exist
    """
    name_sat, beam_idx = sat_beam
    if name_sat not in self.all_sat:
      raise ValueError('Constellation doesn\'t have this satellite')
    self.all_sat[name_sat].set_beamwidth(beam_idx, beamwidth)

  def add_serving(self, sat_bema: SatBeamID, ue_name: str) -> None:
    """Add the serving information

    Args:
        sat_bema (SatBeamID): The index of the serving satellite and beam
        ue_name (str): The name of the users served by this SatBeam
    """
    (sat_name, beam_index) = sat_bema
    self.all_sat[sat_name].add_cell_topo_info(ue_name, beam_index)

  def drop_serving(self, sat_name: str, ue_name: str) -> None:
    """Drop the serving information

    Args:
        sat_name (str): The name of the dat
        ue_name (str): The name of the users
    """
    self.all_sat[sat_name].drop_cell_topo_info(ue_name)

  def assign_training_set(self, sat_name: str, train_set: Set[int]):
    """Assign the training set to the satellite
       (This is dsigned to be used for other training algorithm.
       No need to call this when using 'scan_ues'.)

    Args:
        sat_name (str): The name the satellite
        train_set (Set[int]): training set of the satellite
    """
    self.all_sat[sat_name].assign_train_set(train_set)

  def cal_throughput(self,
                     ues: List[User],
                     mode: str = 'run',
                     sinr: Dict[str, float] | None = None,
                     interference_beams: Set[SatBeamID] | None = None) -> Dict[str, float]:
    """Calculate the throughput of each user

    Args:
        ues (List[User]): The users
        mode (str): The execution mode of this method
                    1. 'run'
                    2. 'debug' (will print detail info)
        sinr (Dict[str, float]): The dictionary of the ue sinr.
        interferece_beams (Set[SatBeamID] | None): custom defined interferece beams

    Returns:
        Dict[str, float]: The throughput of each online user
    """
    throughput = {}
    if sinr is None:
      sinr = self.cal_transmission_sinr(ues=ues,
                                        mode=mode,
                                        interference_beams=interference_beams)
    for ue in ues:
      if ue.online and ue.last_serving:
        sat_name, beam_idx = ue.last_serving
        sat = self.all_sat[sat_name]
        _, _, bandwidth, served_ue_num = sat.get_beam_info(beam_idx)

        if mode == 'debug':
          print(
              f'Bandwidth: {bandwidth}, '
              f'SINR: {sinr[ue.name]} dB, '
              f'Capacity term: {math.log2(1 + util.tolinear(sinr[ue.name]))}, '
              f'channel num: {served_ue_num}, ')

        throughput[ue.name] = (bandwidth *
                               math.log2(1 + util.tolinear(sinr[ue.name])) /
                               served_ue_num)

    return throughput

  def _no_training_result(self, ues: List[User], sat: Satellite) -> Dict[str, List[float]]:
    """When the satellite has no training beam, all the training sinr is 0 (-inf dB). 

    Args:
        ues (List[User]): ue under the satellite
        sat (Satellite): the satellite with no training beam.

    Returns:
        Dict[str, List[float]]: A Dict contains a -inf list of each user.
    """
    ues_sinr = {}
    for ue in ues:
      ues_sinr[ue.name] = [constant.MIN_NEG_FLOAT] * sat.cell_topo.cell_number
    return ues_sinr
