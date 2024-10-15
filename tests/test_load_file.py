
from MA_TD3.misc import misc

f = misc.load_rt_file()

print(f)
agent_name_list = ['3_0_24', '2_0_1', '1_0_9']
for sat_name in agent_name_list:
  for t in (range(50)):
    print(sat_name, t, f[sat_name][t])
