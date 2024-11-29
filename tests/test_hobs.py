import collections
import numpy as np
from low_earth_orbit.util import constant
from low_earth_orbit.channel.channel import Channel
from low_earth_orbit.constellation.constellation import Constellation, ConstellationData


shell_num = 4
plane_num = [1, 1, 1, 1, 1]
sat_per_plane = [22, 22, 20, 58, 43]
sat_height = np.array([550, 540, 570, 560, 560]) * constant.KM
inclination = np.array([53, 53.2, 70, 97.6, 97.6]) * constant.PI_IN_RAD
shell_color = ["r", "#7E2F8E", "#EDB120", "m", "k"]

constel = Constellation(
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

sat_name = '3_0_24'
a = []
for i in range(4):
  beam_index = np.random.randint(0, 19)
  sinr = np.random.randint(8, 20)
  a.append(((sat_name, beam_index), constel.all_sat[sat_name].cell_topo.beam_list[beam_index].center_point, sinr))

serv_hist = collections.deque(a)
print([serv_data[0][1] for serv_data in serv_hist])

succ_index = 0
last_beam_pos = None
for i, serv_data in reversed(list(enumerate(serv_hist))):
  if serv_data[-1] >= constant.SINR_THRESHOLD:
    succ_index = i
    last_beam_pos = serv_data[1]
    break

print(f'succ index: {succ_index}')

long_diff_list = [serv_hist[i][1].geodetic.longitude - serv_hist[i - 1][1].geodetic.longitude
                  for i in range(1, len(serv_hist))]
lati_diff_list = [serv_hist[i][1].geodetic.latitude - serv_hist[i - 1][1].geodetic.latitude
                  for i in range(1, len(serv_hist))]

print(f'long diff list: {long_diff_list}')

s = (constant.DEFAULT_TRAINING_WINDOW_SIZE - succ_index)
epsilon_long = s * max([abs(x) for x in long_diff_list])
epsilon_lati = s * max([abs(x) for x in lati_diff_list])

print(f's: {s}, epsilon_long: {epsilon_long}')
