"""The Cell Topology module."""

from typing import Dict, List, Optional, Set, overload, Tuple, Literal

import math

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from ..util import Position
from ..util import Cartesian
from ..util import constant
from ..util import util
from ..ground_user import User
from .beam import Beam


class CellTopology(object):
  """The cell Topology class.

  Attributes:
    beam_list (list[Beam]): The list of the beams corresponding to the cells
    grid_points (list[Position]): The grid points of the cells
    serving (Dict[str, int]): Storge the name of ue to serving beam_index
  """

  serving: Dict[str, int]
  __training_beam: Set[int]
  grid_points: List[Position]
  center_u: npt.NDArray[np.float64]
  center_v: npt.NDArray[np.float64]
  center_x: npt.NDArray[np.float64]
  center_y: npt.NDArray[np.float64]
  center_z: npt.NDArray[np.float64]
  grid_x: npt.NDArray[np.float64]
  grid_y: npt.NDArray[np.float64]
  grid_z: npt.NDArray[np.float64]

  def __init__(
      self,
      center_point: Position,
      cell_radius: float = constant.DEFAULT_CELL_RADIUS,
      cell_layer: int = constant.DEFAULT_CELL_LAYER,
  ):
    """The __init__ funciton for cell topology.

    Args:
      center_point (Position): The center position of the cell topology
      cell_radius (float): The radius of the cells
      cell_layer (float): The layer of the cell topology

    Raise:
      ValueError: The user must initialize the center position,
                  and the cell radius and the cell layer must be positive.
    """
    self._cell_layer = cell_layer
    self._cell_radius = cell_radius

    self._cell_number = 1
    for i in range(1, self.cell_layer):
      self._cell_number += i * 6

    self._beam_list = [
        Beam(center_point=center_point) for _ in range(self.cell_number)
    ]

    self.generate_xyz_coord_grid()
    self.center_point = center_point

    if cell_radius < 0 or cell_layer < 0:
      raise ValueError(
          "The cell radius and the cell layer must be positive."
      )

    self.serving = {}
    self.__training_beam = set()
    self.__non_training_beam = set(i for i in range(self.cell_number))
    self.grid_points = []

  @property
  def cell_radius(self):
    return self._cell_radius

  @property
  def cell_layer(self):
    return self._cell_layer

  @property
  def cell_distance(self):
    return 1.5 * self.cell_radius / math.cos(constant.PI / 6)

  @property
  def center_point(self):
    return self._center_point

  @center_point.setter
  def center_point(self, center: Position):
    if center is None:
      raise ValueError("center cannot be None")
    self._center_point = center
    self.update_cell_center_pos()

  @property
  def cell_number(self):
    return self._cell_number

  @property
  def beam_list(self):
    return self._beam_list

  @property
  def training_beam(self):
    return self.__training_beam

  @training_beam.setter
  def training_beam(self, beam_set):
    self.__training_beam = beam_set

  @property
  def non_training_beam(self):
    return self.__non_training_beam

  @training_beam.setter
  def non_training_beam(self, beam_set):
    self.__non_training_beam = beam_set

  @property
  def serving_status(self) -> Dict[int, int]:
    """Number of users in each beam.

    Returns:
        Dict[int, int]: Number of users in each beam.
    """
    res: Dict[int, int] = {}

    for beam_index in self.serving.values():
      if beam_index in res:
        res[beam_index] += 1
      else:
        res[beam_index] = 1

    return res

  @property
  def online_beam_set(self) -> Set[int]:
    return set(self.serving.values())

  @property
  def training_beam_num(self) -> float:
    return len(self.__training_beam)

  def clear_training_beam(self):
    self.__training_beam.clear()
    self.non_training_beam = set(i for i in range(self.cell_number))

  def add_training_beam(self, beam_set: Set[int]):
    self.training_beam.update(beam_set)
    self.non_training_beam.difference_update(beam_set)

  def generate_xyz_coord_grid(self):
    """Generate the XYZ-plane coordinate of the cell grid"""

    self.center_u = np.zeros((self.cell_number,))
    self.center_v = np.zeros((self.cell_number,))
    idx = 1
    for i in range(self.cell_layer):
      if i == 0:
        self.center_u[0] = 0
        self.center_v[0] = 0
      else:
        outer_r = i * self.cell_distance / constant.R_EARTH
        outer_vertex_u = outer_r * np.cos(constant.TOPOLOGY_ANGLE)
        outer_vertex_v = outer_r * np.sin(constant.TOPOLOGY_ANGLE)
        for j in range(constant.DEFAULT_SHAPE):
          temp_center_u = np.linspace(
              outer_vertex_u[j],
              outer_vertex_u[j + 1],
              i,
              endpoint=False,
          )
          temp_center_v = np.linspace(
              outer_vertex_v[j],
              outer_vertex_v[j + 1],
              i,
              endpoint=False,
          )
          self.center_u[idx: idx + i] = temp_center_u
          self.center_v[idx: idx + i] = temp_center_v
          idx += i

    self.generate_cell_grid()

    center_theta = np.arcsin(
        np.sqrt(self.center_u**2 + self.center_v**2)
    )
    self.center_x = constant.R_EARTH * self.center_u
    self.center_y = constant.R_EARTH * self.center_v
    self.center_z = constant.R_EARTH * np.cos(center_theta)

  def generate_cell_grid(
      self,
      sat_height: float | None = None,
      cell_range_mode: str = "fixed_radius",
  ):
    """Generate the cell grid in UV-plane then map to the XYZ-plane.

    Args:
        sat_height (float): The height of the satellite
        cell_range_mode (str): The cell plotting mode
    """

    grid_u = np.zeros((self.cell_number, constant.CELL_GRID_RESOLUTION))
    grid_v = np.zeros((self.cell_number, constant.CELL_GRID_RESOLUTION))
    for i in range(self.cell_number):
      r = self.get_plotting_cell_radius(i, sat_height, cell_range_mode)
      norm_r = r / constant.R_EARTH
      grid_u[i, :] = (
          norm_r * np.cos(constant.DRAW_THETA) + self.center_u[i]
      )
      grid_v[i, :] = (
          norm_r * np.sin(constant.DRAW_THETA) + self.center_v[i]
      )

    grid_theta = np.arcsin(np.sqrt(grid_u**2 + grid_v**2))
    self.grid_x = constant.R_EARTH * grid_u
    self.grid_y = constant.R_EARTH * grid_v
    self.grid_z = constant.R_EARTH * np.cos(grid_theta)

  def get_plotting_cell_radius(
      self, beam_idx: int, sat_height: float | None, cell_range_mode: str
  ) -> float:
    """Get the cell radius

    Args:
        beam_idx (int): beam index corresponding to the cell
        sat_height (float | None): The height of this satellite
        cell_range_mode (str): plotting mode

    Raises:
        ValueError: The sat_height cannot be none in \'3dB_range\' mode.
        ValueError: The sat_height cannot be none in \'main_lobe_range\' mode.
        ValueError: Plotting mode doesn\'t exist.

    Returns:
        float: The cell radius
    """
    if cell_range_mode == "fixed_radius":
      r = self.cell_radius
    elif cell_range_mode == "3dB_range":
      if sat_height is None:
        raise ValueError(
            f"Must set the sat_height in {cell_range_mode} plotting mode."
        )
      r = (sat_height - constant.R_EARTH) * math.tan(
          self.beam_list[beam_idx].beamwidth_3db
      )
    elif cell_range_mode == "main_lobe_range":
      if sat_height is None:
        raise ValueError(
            f"Must set the sat_height in {cell_range_mode} plotting mode."
        )
      r = (
          (sat_height - constant.R_EARTH)
          * math.tan(self.beam_list[beam_idx].beamwidth_3db)
          * constant.MAIN_LOBE_RANGE
      )
    else:
      raise ValueError(f"No {cell_range_mode} cell plotting mode.")

    return r

  @overload
  def mapping_rotate(self, x: int, y: int, z: int) -> Tuple[int, int, int]:
    ...

  @overload
  def mapping_rotate(
      self,
      x: npt.NDArray[np.float64],
      y: npt.NDArray[np.float64],
      z: npt.NDArray[np.float64],
  ) -> Tuple[
      npt.NDArray[np.float64],
      npt.NDArray[np.float64],
      npt.NDArray[np.float64],
  ]:
    ...

  def mapping_rotate(self, x, y, z):
    """Rotate the XYZ coordinate to the corresponding position of satellite

    Args:
        x (int or npt.NDArray[np.float64]): X coordinate
        y (int or npt.NDArray[np.float64]): Y coordinate
        z (int or npt.NDArray[np.float64]): Z coordinate

    Returns:
        int or npt.NDArray[np.float64]: The rotated coordinate
    """

    # Rotate along the y-axis
    longitude, latitude, _ = self.center_point.geodetic.get()
    rotated_latitude = constant.PI / 2 - latitude * constant.PI_IN_RAD
    rotated_longitude = longitude * constant.PI_IN_RAD

    r1_z = z * np.cos(rotated_latitude) - x * np.sin(rotated_latitude)
    r1_x = z * np.sin(rotated_latitude) + x * np.cos(rotated_latitude)
    r1_y = y

    # Rotate along the z-axis
    r2_x = r1_x * np.cos(rotated_longitude) - r1_y * np.sin(
        rotated_longitude
    )
    r2_y = r1_x * np.sin(rotated_longitude) + r1_y * np.cos(
        rotated_longitude
    )
    r2_z = r1_z

    return r2_x, r2_y, r2_z

  def update_cell_center_pos(self):
    """Update beam center position based on the center point."""

    center_x, center_y, center_z = self.mapping_rotate(
        self.center_x, self.center_y, self.center_z
    )

    for i in range(self.cell_number):
      self.beam_list[i].center_point = Position(
          cartesian=Cartesian(x=center_x[i], y=center_y[i], z=center_z[i])
      )

  def update_grid_pos(self):
    """Update grid point position based on the center point."""

    grid_x, grid_y, grid_z = self.mapping_rotate(
        self.grid_x, self.grid_y, self.grid_z
    )

    self.grid_points = [
      Position(cartesian=Cartesian(x=x, y=y, z=z))
      for x, y, z in zip(
          grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)
      )
    ]

  def plot_cell(
      self, ax: plt.Axes, cell_plot_mode: str, color_dict: Dict[str, str]
  ):
    """Plot each cell in the topology."""

    long_list = [pos.geodetic.longitude for pos in self.grid_points]
    lati_list = [pos.geodetic.latitude for pos in self.grid_points]

    for cell_i in range(self.cell_number):
      index_start = constant.CELL_GRID_RESOLUTION * cell_i
      index_end = constant.CELL_GRID_RESOLUTION * (cell_i + 1)

      c = self.get_cell_color(
          cell_i=cell_i,
          cell_plot_mode=cell_plot_mode,
          color_dict=color_dict,
      )

      if c is not None:
        ax.plot(
            long_list[index_start:index_end],
            lati_list[index_start:index_end],
            c=c,
            alpha=constant.CELL_ALPHA,
        )

  def get_cell_color(
      self,
      cell_i: int,
      cell_plot_mode: Literal["all", "active_only", "active_and_training"],
      color_dict: Dict[str, str],
  ) -> str | None:
    """Get the plotting color of the cell.

    Args:
        cell_i (int): The index of the cell
        cell_plot_mode (str): The plotting mode

    Returns:
        str | None: The plotting color of the cell
    """
    if cell_i in self.serving.values():
      return color_dict.get("serv_cell", constant.DEFAULT_CELL_SERV_COLOR)
    if cell_plot_mode == "active_only":
      return None

    if cell_i in self.__training_beam:
      return color_dict.get("scan_cell", constant.DEFAULT_CELL_SCAN_COLOR)
    if cell_plot_mode == "active_and_training":
      return None

    if cell_plot_mode == "all":
      return color_dict.get("norm_cell", constant.DEFAULT_CELL_NORM_COLOR)
    return None

  def plot_topo_center(
      self,
      ax: plt.Axes,
      color_dict: Dict[str, str],
  ):
    """Plot center point of the topology."""

    center_longitude, center_latitude, _ = self.center_point.geodetic.get()
    c = color_dict.get("topo_center", constant.DEFAULT_TOPO_CENTER_COLOR)
    ax.scatter(
        center_longitude, center_latitude, s=constant.SAT_MARKER_SIZE, c=c
    )

  def plot_cell_center(self, ax: plt.Axes):
    """Plot center point of each cell in the topology."""

    for beam in self.beam_list:
      (
          center_longitude,
          center_latitude,
          _,
      ) = beam.center_point.geodetic.get()
      ax.scatter(center_longitude, center_latitude)

  def plot_geodetic_cell_topology(
      self,
      ax: plt.Axes,
      sat_height: float | None = None,
      cell_range_mode: Literal[
          "fixed_radius", "3dB_range", "main_lobe_range"
      ] = "fixed_radius",
      cell_plot_mode: Literal[
          "all", "active_only", "active_and_training"
      ] = "all",
      color_dict: Dict[
          Literal["topo_center", "serv_cell", "scan_cell", "norm_cell"], str
      ] = None,
  ):
    """Plot the cell topology in the geodetic coordinate

    Args:
        ax (plt.Axes): The axes of the plot
        sat_height (float | None, optional): The height of the corresponding satellite. Defaults to None.
        cell_range_mode (str, optional): Defaults to 'fixed radius'. There is three modes.\n
                                         1. 'fixed_radius'
                                         2. '3dB_range'
                                         3. 'main_lobe_range'
        cell_plot_mode (str, optional): Defaults to 'all'. There is three modes.\n
                                        1. 'all'
                                        2. 'active_only'
                                        3. 'active_and_training'
        color_dict (Dict[str, str]): The color dict you can specify the color of the cell plot.
                                     It will use default color if you don't assign the color.\n
                                     1. 'topo_center': the color of the topology center (satellite)
                                     2. 'serv_cell': the color of the serving cells
                                     3. 'scan_cell': the color of the cells perform beam training
                                     4. 'norm_cell': the color of the normal cells

    Raises:
        ValueError: No ax is given.
    """

    if ax is None:
      raise ValueError("Must give the axes of the plot")

    if color_dict is None:
      color_dict = {}

    if cell_range_mode != "fixed radius":
      self.generate_cell_grid(
          sat_height=sat_height, cell_range_mode=cell_range_mode
      )

    self.update_grid_pos()

    self.plot_topo_center(ax, color_dict)
    # self.plot_cell_center(ax)

    self.plot_cell(ax, cell_plot_mode, color_dict)

  def sinr_of_users(
      self,
      serving_ue: List[User],
      tx_gain: List[float],
      channel_loss: List[float],
      i_power: List[float],
      mode: str = "run",
  ) -> List[float]:
    """Get the sinr of a list of user

    Args:
        ue (User): the user
        tx_gain (float): transmitting antenna gain
        channel_loss (float): the channel loss from the ue to sat
        i_power (float): The total interference power each ue gets
        mode (str): the mode this function is running
                    (run or debug)

    Returns:
        float: the SINR
    """
    return [
        self.beam_list[self.serving[ue.name]].calc_sinr(
            ue=ue,
            tx_gain=tx_gain[i],
            channel_loss=channel_loss[i],
            interference_power=i_power[i],
            mode=mode,
        )
        for i, ue in enumerate(serving_ue)
    ]

  def beam_pos_of_serving_ue(self, ue: User) -> Position:
    """Return the position of the beam which is serving the ue

    Args:
        ue (User): The user

    Returns:
        Position: The position of the beam center
    """
    return self.beam_list[self.serving[ue.name]].center_point

  def find_nearby(
      self, ue: User, r: Optional[float] = None
  ) -> Set[int]:
    """Find the nearby beam_index within r.

    Args:
        ue_pos: The position of the ue.
        r: The searching radius.

    Returns:
        The list of index of the beam index.
    """
    # James
    search_r = 1.5

    if not isinstance(r, float):
      r = self.cell_radius * search_r

    def _dis(beam_pos: Position, ue_pos: Position, r: float):
      dis = beam_pos.calculate_distance(ue_pos)
      return dis <= r

    res = set(
        i
        for i, item in enumerate(self.beam_list)
        if _dis(item.center_point, ue.position, r)
    )
    return res

  def hobs(self, ue: User):
    serv_hist = ue.serving_history
    succ_index = 0
    last_beam_pos = None
    for i, serv_data in reversed(list(enumerate(serv_hist))):
      if serv_data[-1] >= constant.SINR_THRESHOLD:
        succ_index = i
        last_beam_pos = serv_data[1]
        break

    if last_beam_pos is None:
      return set(i for i in range(self.cell_number))

    long_diff_list = [serv_hist[i][1].geodetic.longitude - serv_hist[i - 1][1].geodetic.longitude
                      for i in range(1, len(serv_hist))]
    lati_diff_list = [serv_hist[i][1].geodetic.latitude - serv_hist[i - 1][1].geodetic.latitude
                      for i in range(1, len(serv_hist))]

    s = (constant.DEFAULT_TRAINING_WINDOW_SIZE - succ_index)
    epsilon_long = s * max([abs(x) for x in long_diff_list])
    epsilon_lati = s * max([abs(x) for x in lati_diff_list])

    res = self._get_training_area()
    return res

  def _get_training_area(self, long_start, long_end, lati_start, lati_end):
    res = set()
    for beam_idx in self.non_training_beam:
      beam_long = self.beam_list[beam_idx].center_point.geodetic.longitude
      beam_lati = self.beam_list[beam_idx].center_point.geodetic.latitude
      if (beam_long >= long_start and beam_long <= long_end
              and beam_lati >= lati_start and beam_lati <= lati_end):
        res.update(beam_idx)

    return res

  def set_beam_power(self, beam_idx: int, tx_power: float):
    """Set the tx power of the beam

    Args:
        beam_idx (int): beam index
        tx_power (float): transmitting power in dBm
    """
    self.beam_list[beam_idx].tx_power = tx_power

  def set_beamwidth(self, beam_idx: int, beamwidth: float):
    """Set the 3dB beamwidth of the beam

    Args:
        beam_idx (int): beam index
        beamwidth (float): 3dB beamwidth
    """
    self.beam_list[beam_idx].beamwidth_3db = beamwidth

  def clear_power(self):
    """Set all the beam power to -inf (dB)"""
    for beam in self.beam_list:
      beam.tx_power = constant.MIN_NEG_FLOAT

  def add_serving(self, ue_name: str, beam_idx: int):
    """Update the serving dict in both cell topo and beam

    Args:
        ue_name (str): The user
        beam_idx (int): The serving beam
    """
    self.serving[ue_name] = beam_idx
    self.beam_list[beam_idx].add_serving(ue_name)

  def remove_serving(self, ue_name: str):
    """Remove the serving information in both cell topo and beam

    Args:
        ue_name (str): The user
    """
    beam_idx = self.serving.pop(ue_name, None)

    if beam_idx is not None:
      self.beam_list[beam_idx].remove_serving(ue_name)
    else:
      # lgo error in the future
      print("No such beam_idx, error!")

  def all_beam_power(self) -> float:
    """Tx power of all beams

    Returns:
        float: The total beam power
    """
    power_mw = sum(util.tolinear(beam.tx_power) for beam in self.beam_list)
    return util.todb(power_mw)

  def print_all_beams(self):
    for i, beam in enumerate(self.beam_list):
      print(f"beam: {i}, {beam}")

  def get_beam_info(self, beam_idx: int) -> Tuple[float, float, float, int]:
    """Get the info of the beam

    Args:
        beam_idx (int): The indx of the beam

    Returns:
        Tuple[float, float, float]: tx_power, central_frequency, bandwidth, served ue number
    """
    beam = self.beam_list[beam_idx]
    return (
        beam.tx_power,
        beam.central_frequency,
        beam.bandwidth,
        len(beam.served_ue),
    )

  def cal_two_beam_dis(self, beam_one_i: int, beam_two_i: int):
    beam_one = self.beam_list[beam_one_i].center_point.geodetic
    beam_two = self.beam_list[beam_two_i].center_point.geodetic
    d_long, d_lati, _ = beam_one.pos_different(beam_two)
    return d_long, d_lati

  '''def cal_graph_beam_dis(
      self, scale: tuple[float, float], beam_link=constant.BEAM_NEIGHBOR
  ):
    res = []
    for cell_list in beam_link:
      temp_list = [0] * self.cell_number
      for baem_one, beam_two in cell_list:
        d_long, d_lati = self.cal_two_beam_dis(baem_one, beam_two)
        s_long, s_lati = scale
        inner = s_long * d_long + s_lati * d_lati
        temp_list[beam_two] = np.maximum(0, inner)
      res.append(temp_list)
    return np.array(res)
  '''
