"""The basic util module."""

import os
from typing import Tuple, overload

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
