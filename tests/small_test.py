import time

import matplotlib.pyplot as plt
import numpy as np

from low_earth_orbit.constellation import Constellation
from low_earth_orbit.constellation import ConstellationData
from low_earth_orbit.util import constant
from low_earth_orbit.util import Position
from low_earth_orbit.util import Geodetic
from low_earth_orbit.ground_user import User
from low_earth_orbit.channel import Channel
from low_earth_orbit import util
from low_earth_orbit.nmc import NMC
from MA_TD3.misc import misc

shell_num = 4
plane_num = [1, 1, 1, 1, 1]
sat_per_plane = [22, 22, 20, 58, 43]
sat_height = np.array([550, 540, 570, 560, 560]) * constant.KM
inclination = np.array([53, 53.2, 70, 97.6, 97.6]) * constant.PI_IN_RAD
shell_color = ["r", "#7E2F8E", "#EDB120", "m", "k"]

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
    channel=Channel()
)

simulation_time = 50

fig = plt.figure(figsize=(16, 9), dpi=80)
fig.set_tight_layout(True)
ax = plt.axes()
ax.set_aspect("equal", adjustable="box")
plt.ion()

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
sat_name_list = ['3_0_24']  # ['3_0_24', '2_0_1', '1_0_9']
r = 5

f = misc.load_rt_file('rotated_rt_result')

path_loss_array = np.asarray([data['path loss (dB)']
                              for t in f
                              for sat_name in f[t]
                              for b_i in f[t][sat_name]
                              for data in f[t][sat_name][b_i]])
print(len(path_loss_array))

long = constant.ORIGIN_LONG
lati = constant.ORIGIN_LATI
LEO_constellation.update_sat_position(time=-20 * constant.TIMESLOT)
for t in range(simulation_time + 1):
  # _time = time.time()

  ax.clear()
  util.plot_taiwan_shape(ax)
  for ue in user_list:
    ax.scatter(
        ue.position.geodetic.longitude,
        ue.position.geodetic.latitude,
        s=constant.UE_MARKER_SIZE,
        c="g",
    )

  for sat in LEO_constellation.all_sat.values():
    if sat.name not in sat_name_list:
      continue
    diff_la = abs(sat.position.geodetic.latitude - lati)
    diff_lo = abs(sat.position.geodetic.longitude - long)
    if diff_la < r and diff_lo < r:
      ax.text(
          sat.position.geodetic.longitude,
          sat.position.geodetic.latitude,
          sat.name,
      )
      sat.cell_topo.plot_geodetic_cell_topology(
          ax=ax,
          sat_height=sat.position.geodetic.height,
          cell_range_mode="main_lobe_range",
          cell_plot_mode="all",
          color_dict={"topo_center": "y", "scan_cell": "b"},
      )
      for b_i in range(sat.cell_topo.cell_number):
        beam_pos = sat.cell_topo.beam_list[b_i].center_point
        pl = min([data['path loss (dB)'] for data in f[t][sat.name][b_i]])
        ax.text(beam_pos.geodetic.longitude,
                beam_pos.geodetic.latitude,
                f'{b_i}: {pl}',
                fontsize=10,
                horizontalalignment='center',
                verticalalignment='center',)
  plt.xlim((long - r, long + r))
  plt.ylim((lati - r, lati + r))
  plt.show()
  if t < 3:
    plt.pause(0.2)
  else:
    plt.pause(15)

  LEO_constellation.scan_ues(user_list)
  # print(f'---{time.time() - _time}')
  # print(obj.servable)
  LEO_constellation.update_sat_position(time=constant.TIMESLOT)
  obj_nmc.a3_event_check()
  throughput = LEO_constellation.cal_throughput(ues=user_list)
  for key, value in throughput.items():
    print(f"{key}: {value:.1e}", end="  ")
  print("\n")
  print(f"t: {t}")
  # print(time.time() - _time)
