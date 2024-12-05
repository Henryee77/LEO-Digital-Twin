
import matplotlib.pyplot as plt
from low_earth_orbit.nmc import NMC
from low_earth_orbit import util
from low_earth_orbit.channel.channel import Channel
from low_earth_orbit.ground_user import User
from low_earth_orbit.util import Geodetic
from low_earth_orbit.util import Position
from low_earth_orbit.util import constant
from low_earth_orbit.constellation import ConstellationData
from low_earth_orbit.constellation import Constellation
from MA_TD3.agent import Agent
from MA_TD3.misc.replay_buffer import ReplayBuffer
from low_earth_orbit.util import util, constant
from MA_TD3.misc import misc
import argparse
import functools
import time
import timeit
import copy
import numpy as np
from random import sample


def init_constel():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument(
      '--beam-sweeping-mode', type=str, default='ABS',
      help='Beam-sweeping mode')
  parser.add_argument(
      '--cell-layer-num', type=int, default=constant.DEFAULT_CELL_LAYER,
      help='Number of cell layer (related to cell number)')
  parser.add_argument(
      '--leo-computaion-speed', type=float, default=constant.DEFAULT_LEO_CPU_CYCLE,
      help='LEO computation speed')
  args = parser.parse_args()
  shell_num = 4
  plane_num = [1, 1, 1, 1, 1]
  sat_per_plane = [22, 22, 20, 58, 43]
  sat_height = np.array([550, 540, 570, 560, 560]) * constant.KM
  inclination = np.array([53, 53.2, 70, 97.6, 97.6]) * constant.PI_IN_RAD

  LEO_constellation = Constellation(
      setup_data_list=[
          ConstellationData(
              plane_num=plane_num[i],
              sat_per_plane=sat_per_plane[i],
              sat_height=sat_height[i],
              inclination_angle=inclination[i],
          )
          for i in range(shell_num)
      ],
      args=args
  )
  return LEO_constellation


def test_leo_sim(sat_name_list):

  LEO_constellation = init_constel()

  simulation_time = 50

  ue_long = [120.99, 121.58, 120.74]
  ue_lati = [24.78, 25.03, 22.39]

  user_list = [
      User(
          f"ue{i}",
          Position(
              geodetic=Geodetic(
                  longitude=ue_long[i],
                  latitude=ue_lati[i],
                  height=constant.R_EARTH,
              )
          ),
      )
      for i in range(len(ue_long))
  ]

  obj_nmc = NMC(constellation=LEO_constellation, ues=user_list)

  LEO_constellation.update_sat_position(time=-10 * constant.MOVING_TIMESLOT)
  for t in range(simulation_time + 1):

    print(f'---t:{t}---')
    start_time = time.time()
    LEO_constellation.scan_ues(user_list, sat_name_list=sat_name_list, scan_mode='SCBS')
    print(f'scan_ues:---{time.time() - start_time}')

    start_time = time.time()
    LEO_constellation.update_sat_position(time=constant.MOVING_TIMESLOT)
    # print(f'---{time.time() - start_time}')

    start_time = time.time()
    obj_nmc.a3_event_check()
    # print(f'---{time.time() - start_time}')

    start_time = time.time()
    obj_nmc.update_ues_serving_history()
    # print(f'---{time.time() - start_time}')

    start_time = time.time()
    throughput = LEO_constellation.cal_throughput(ues=user_list)
    # print(f'{throughput}---{time.time() - start_time}')


def test_export_power_dict(constell: Constellation, sat_name_list):
  temp_power_dict = {}
  for sat_name in sat_name_list:
    temp_power_dict[sat_name] = constell.all_sat[sat_name].export_power_dict()

  for sat_name in sat_name_list:
    constell.all_sat[sat_name].import_power_dict(temp_power_dict[sat_name])


def test_func():
  pass


if __name__ == '__main__':
  constell = init_constel()
  sat_name_list = ['3_0_24', '2_0_1', '1_0_9']

  # print(timeit.timeit(functools.partial(test_export_power_dict, constell, sat_name_list), number=round(1e2)))
  print(timeit.timeit(functools.partial(test_func), number=round(1e1)))
