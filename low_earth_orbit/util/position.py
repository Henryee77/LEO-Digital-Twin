"""The position module."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional
from typing import Tuple
import numpy as np

from . import constant


@dataclass
class Cartesian:
  """The Cartesian coordinates."""
  x: float
  y: float
  z: float

  def __str__(self) -> str:
    return f'x: {self.x}, y: {self.y}, z: {self.z}'

  def get(self) -> Tuple[float, float, float]:
    return self.x, self.y, self.z

  def to_geodetic(self) -> Geodetic:
    """Change the Cartesian to Geodetic."""
    height = math.sqrt(self.x**2 + self.y**2 + self.z**2)
    xy = math.sqrt(self.x**2 + self.y**2)
    latitude = math.atan2(self.z, xy) / constant.PI_IN_RAD
    longitude = math.atan2(self.y, self.x) / constant.PI_IN_RAD

    return Geodetic(longitude, latitude, height)


@dataclass
class Geodetic:
  """The Geodetic coordinates."""
  longitude: float
  latitude: float
  height: float

  def __str__(self) -> str:
    return (f'longitude: {self.longitude}, '
            f'latitude: {self.latitude}, '
            f'height: {self.height}')

  def get(self) -> Tuple[float, float, float]:
    return self.longitude, self.latitude, self.height

  def to_cartesian(self) -> Cartesian:
    """Change the Geodetic to Cartesian."""
    long_r = self.longitude * constant.PI_IN_RAD
    lati_r = self.latitude * constant.PI_IN_RAD
    x = self.height * math.cos(lati_r) * math.cos(long_r)
    y = self.height * math.cos(lati_r) * math.sin(long_r)
    z = self.height * math.sin(lati_r)

    return Cartesian(x, y, z)


@dataclass
class Orbital:
  """The Orbital coordinates."""
  inclination: float
  small_omega: float
  large_omega: float
  radius: float

  def __str__(self) -> str:
    return (f'inclination: {self.inclination}, '
            f'small_omega: {self.small_omega}, '
            f'large_omega: {self.large_omega}, '
            f'radius: {self.radius}')

  def get(self) -> Tuple[float, float, float, float]:
    return self.inclination, self.small_omega, self.large_omega, self.radius

  def to_cartesian(self) -> Cartesian:
    """Change the Orbital to Cartesian."""
    sin_o_cos_i = math.sin(self.small_omega) * math.cos(self.inclination)

    x = self.radius * (math.cos(self.large_omega) * math.cos(self.small_omega) -
                       math.sin(self.large_omega) * sin_o_cos_i)
    y = self.radius * (math.sin(self.large_omega) * math.cos(self.small_omega) +
                       math.cos(self.large_omega) * sin_o_cos_i)
    z = self.radius * math.sin(self.small_omega) * math.sin(self.inclination)

    return Cartesian(x, y, z)

  def to_geodetic(self) -> Geodetic:
    """Change the Orbital to Geodetic."""
    return self.to_cartesian().to_geodetic()


class Position(object):
  """The position class."""

  def __init__(
      self,
      cartesian: Optional[Cartesian] = None,
      geodetic: Optional[Geodetic] = None,
      orbital: Optional[Orbital] = None,
  ):
    """The __init__ funciton for position.

    Args:
        cartesian (Optional[Cartesian]): The longitude of the object.
        geodetic (Optional[Geodetic]): The latitude of the object.

    Raise:
        ValueError: The user must select cartesian or geodetic as the inout.
    """

    if cartesian:
      self.cartesian = cartesian
    elif geodetic:
      self.geodetic = geodetic
    elif orbital:
      self.orbital = orbital
    else:
      raise ValueError('You must select cartesian or geodetic as the input.')

  @property
  def cartesian(self):
    return self._cartesian

  @cartesian.setter
  def cartesian(self, coordinate: Cartesian):
    self._cartesian = coordinate
    self._geodetic = coordinate.to_geodetic()

  @property
  def geodetic(self):
    return self._geodetic

  @geodetic.setter
  def geodetic(self, coordinate: Geodetic):
    self._geodetic = coordinate
    self._cartesian = coordinate.to_cartesian()

  @property
  def orbital(self):
    if getattr(self, '_orbital', None) is None:
      raise ValueError('You haven\'t initialize the Orbital.')
    return self._orbital

  @orbital.setter
  def orbital(self, coordinate: Orbital):
    self._orbital = coordinate
    self._geodetic = coordinate.to_geodetic()
    self._cartesian = coordinate.to_cartesian()

  def __eq__(self, target: Position):
    """Define the equal operator"""
    return self.cartesian.get() == target.cartesian.get()

  def calculate_distance(self, target: Position) -> float:
    """Calculate distance and angel.

    Args:
        target (Position): The target that we want to Calculate.

    Returns:
        The distance between two position
    """
    diff_x = target.cartesian.x - self.cartesian.x
    diff_y = target.cartesian.y - self.cartesian.y
    diff_z = target.cartesian.z - self.cartesian.z
    dis = math.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
    return dis

  def angle_between_targets(self, target1: Position,
                            target2: Position) -> float:
    """Angle between two lines (self-target1 and self-target2)

    Args:
        target1 (Position): The position of target1
        target2 (Position): The position of target2

    Returns:
        float: The angle in radian
    """
    d1 = self.calculate_distance(target1)
    d2 = self.calculate_distance(target2)
    d3 = target1.calculate_distance(target2)
    theta_temp = (d1**2 + d2**2 - d3**2) / (2 * d1 * d2)
    return math.acos(np.clip(theta_temp, -1, 1))

  def cal_elevation_angle(self, target: Position) -> float:
    """Calculate the elevation angle

    Args:
        target (Position): The ground point

    Returns:
        float: The angle in radian
    """
    earth_center = Position(Cartesian(x=0, y=0, z=0))
    return target.angle_between_targets(self, earth_center) - constant.PI / 2
