"""The basic util module."""

import os
from typing import Tuple, Set, overload

import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from . import constant

DIRPATH = os.path.dirname(__file__)


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
  return 10 * np.log10(linear)


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


def rescale_value(x, r_min, r_max, t_min, t_max) -> float:
  """Rescale the value from domain R to doamin T

  Args:
      x (_type_): The input value
      r_min (_type_): min value of space R
      r_max (_type_): max value of space R
      t_min (_type_): min value of space T
      t_max (_type_): max value of space T

  Returns:
      float: The rescaled value
  """
  res = (x - r_min) / (r_max - r_min) * (t_max - t_min) + t_min
  # print(f'debug: {x}, {res}, {t_min}, {t_max}')
  return res


def truncate(x: float, precision: int = 3) -> float:
  mult = 10 ** (precision)
  return float(math.floor(x * mult)) / mult


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
