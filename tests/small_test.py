
from MA_TD3.agent import Agent
from MA_TD3.misc.replay_buffer import ReplayBuffer
from low_earth_orbit.util import util
import math
import time
from random import sample

start_time = time.time()

a = [i for i in range(6)]
idx = [1, 2]
print(a[idx])

print(time.time() - start_time)
