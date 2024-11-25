
from MA_TD3.agent import Agent
from MA_TD3.misc.replay_buffer import ReplayBuffer
from low_earth_orbit.util import util, constant
import math
import time
from random import sample

start_time = time.time()

print(type(constant.MOVING_TIMESLOT))
print(2 - constant.MOVING_TIMESLOT)

print(time.time() - start_time)
