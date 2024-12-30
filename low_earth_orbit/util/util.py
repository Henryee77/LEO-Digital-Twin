"""The basic util module."""
from scipy import special as sp
import os
from typing import overload, Tuple, Set, Dict, Any

import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from . import constant

DIRPATH = os.path.dirname(__file__)


def qfunc(x):
  return 0.5 - 0.5 * sp.erf(x / math.sqrt(2))


@overload
def todb(linear: float) -> float:
  ...


@overload
def todb(linear: npt.NDArray) -> npt.NDArray:
  ...


def todb(linear):
  """Change the linear to db.

  Args:
      linear: In linear scale.

  Returns:
      In db scale.
  """
  return np.clip(10 * np.log10(linear), constant.MIN_DB, constant.MAX_DB)


@overload
def tolinear(db: float) -> float:
  ...


@overload
def tolinear(db: npt.NDArray) -> npt.NDArray:
  ...


def tolinear(db):
  """Change the dB to linear.

  Args:
      db: In dB scale.

  Returns:
      In linear scale.
  """
  return 10**(db / 10)


def sign(x: float) -> int:
  if x >= 0:
    return 1
  if x < 0:
    return -1
  raise ValueError("NAN")


@overload
def rescale_value(x: npt.NDArray, r_min: float, r_max: float, t_min: float, t_max: float) -> npt.NDArray:
  ...


def rescale_value(x: float, r_min: float, r_max: float, t_min: float, t_max: float) -> float:
  """Rescale the value from domain R to doamin T

  Args:
      x (float): The input value
      r_min (float): min value of space R
      r_max (float): max value of space R
      t_min (float): min value of space T
      t_max (float): max value of space T

  Returns:
      float: The rescaled value
  """
  res = (x - r_min) / (r_max - r_min) * (t_max - t_min) + t_min
  # print(f'debug: {x}, {res}, {t_min}, {t_max}')
  return res


@overload
def standardize(x: npt.NDArray, mean: float, stdv: float) -> npt.NDArray:
  ...


def standardize(x: float, mean: float, stdv: float) -> float:
  """Standardization

  Args:
      x (float): input
      mean (float): mean
      stdv (float): standard deviation

  Returns:
      float: (x - mean) / stdv
  """
  return (x - mean) / stdv


def truncate(x: float, precision: int = 3) -> float:
  mult = 10 ** (precision)
  return float(math.floor(x * mult)) / mult


def random_sign():
  """Uniformly generate {-1, 1} 

  Returns:
      int: sign
  """
  return np.random.choice([-1, 1])


def calc_sat_angular_speed(radius: float) -> float:
  """Calculate the angular speed of the satellite

  Args:
      radius (float): The radius of the satellite
      in the Earth-center-Eath-fixed coordinate

  Returns:
      The angular speed (rad/s)
  """
  return math.sqrt(constant.STAND_GRAVIT_PARA) / radius**(3. / 2)


def get_taiwan_shape() -> Tuple[npt.NDArray, npt.NDArray]:
  """Read back the Taiwan shape.

  Returns:
    1. The longitude
    2. The latitude
  """
  shape = np.genfromtxt(f'{DIRPATH}/Taiwan.csv', delimiter=',', skip_header=1)
  shape = shape.transpose()
  return shape[1], shape[2]


def plot_taiwan_shape(ax: plt.Axes):
  long, lati = get_taiwan_shape()
  ax.scatter(long, lati, s=1)


def propagation_delay(distance) -> float:
  return distance / constant.LIGHT_SPEED


def rt_delay(ray_spacing: float, unit_num, comp_speed) -> float:
  return unit_num * constant.RT_COMP_SIZE * (180 / ray_spacing) / comp_speed


def d_longitude(origin_latitude: float, distance: float) -> float:
  return (distance / constant.R_EARTH) / constant.PI_IN_RAD / math.cos(origin_latitude * constant.PI_IN_RAD)


def d_latitude(distance: float) -> float:
  return (distance / constant.R_EARTH) / constant.PI_IN_RAD


def avg_time_sat_dict(dict_2d: Dict[Any, Dict[Any, int | float]]) -> float:
  total_value = 0
  for dict in dict_2d.values():
    total_value += sum(dict.values())
  return total_value / len(dict_2d.keys())
