
from MA_TD3.misc import misc
import numpy as np
from low_earth_orbit.util import constant
from low_earth_orbit.util import util

f = misc.load_rt_file('ue6_rt_result')

path_loss_array = np.asarray([data['path loss (dB)']
                              for t in f
                              for sat_name in f[t]
                              for b_i in f[t][sat_name]
                              for data in f[t][sat_name][b_i]])
print(len(path_loss_array), path_loss_array)
print(np.mean(path_loss_array), np.std(path_loss_array))
print(util.standardize(path_loss_array, np.mean(path_loss_array), np.std(path_loss_array)))


'''agent_name_list = ['3_0_24', '2_0_1', '1_0_9']
for sat_name in agent_name_list:
  for t in (range(50)):
    print(t, sat_name, f[t][sat_name], type(f[t][sat_name]))'''
