"""constellation.py"""

import math
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
      channel: Channel,
  ):
    """The constructer of Constellation Class.

    Args:
        setup_data_list (list[ConstellationData]): All data for constellation.
        channel (Channel): Wireless channel.
    """
    self.shell_num = len(setup_data_list)
    self.setup_data_list = setup_data_list
    self.sat_number = sum(
        data.plane_num * data.sat_per_plane for data in self.setup_data_list)
    self.angular_speed_orbital = [
        util.calc_sat_angular_speed(data.sat_height + constant.R_EARTH)
        for data in self.setup_data_list
    ]
    self.all_sat = {}
    self.wireless_channel = channel
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
                        cell_topo=CellTopology(center_point=Position(
                            geodetic=projection_point)),
                        channel=self.wireless_channel)

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

  def scan_ues(self, ues: List[User], sat_name_list: List[str] = None) -> Dict[str, Dict[str, List[float]]]:
    """Select the training beam and calculate the RSRP,
       and add the servable data to the user

    Args:
        ues (List[User]): The users
    """
    for ue in ues:
      ue.servable_clear()

    if sat_name_list is None:
      sat_name_list = self.all_sat.keys()

    sat_ues_sinr = {}
    for sat_name in sat_name_list:
      sat_ues_sinr[sat_name] = self.all_sat[sat_name].select_train_by_topo(ues)

    return sat_ues_sinr

  def cal_transmission_sinr(self,
                            ues: List[User],
                            mode: str = 'run',
                            interference_beams: Set[SatBeamID] | None = None) -> Dict[str, float]:
    """Calculate the SINR of users.

    This function is only supposed to be used when calculating the throughput
    of data trasmission. Should not be used in the training phase

    Args:
        ues (List[User]): users
        mode (str): The execution mode of this method
                    1. 'run'
                    2. 'debug' (will print detail info)
        interferece_beams (Set[SatBeamID] | None): custom defined interferece beams

    Returns:
        Dict[str, float]: SINR of each user
    """
    last_serving_set = set(ue.last_serving for ue in ues)
    if interference_beams is not None:
      last_serving_set = last_serving_set.union(interference_beams)
    sinr = {}
    for ue in ues:
      i_power = constant.MIN_POSITIVE_FLOAT  # in linear
      for history_data in last_serving_set:
        if history_data is None:
          continue

        last_serving = ue.last_serving

        if last_serving == history_data:
          continue
        sat_name, beam_id = history_data
        sat = self.all_sat[sat_name]
        beam_pos = sat.cell_topo.beam_list[beam_id].center_point
        i_power += util.tolinear(sat.cal_rsrp_one_beam(beam_pos=beam_pos,
                                                       beam_index=beam_id,
                                                       ue=ue))

      best_record = ue.last_serving
      if best_record is None:
        sinr[ue.name] = math.nan
      else:
        best_sat = self.all_sat[best_record[0]]
        sinr[ue.name] = best_sat.sinr_of_users(
            serving_ue=[ue],
            i_power=[util.todb(i_power)],
            mode=mode,
        )[0]

    return sinr

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
              f'Training latency: {sat.training_latency}, '
              f'Bandwidth: {bandwidth}, '
              f'SINR: {sinr[ue.name]}, '
              f'Capacity term: {math.log2(1 + util.tolinear(sinr[ue.name]))}, '
              f'channel num: {served_ue_num}, ')

        throughput[ue.name] = (bandwidth *
                               math.log2(1 + util.tolinear(sinr[ue.name])) /
                               served_ue_num)

    return throughput
