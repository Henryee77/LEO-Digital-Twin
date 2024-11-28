
from MA_TD3.agent import Agent
from MA_TD3.misc.replay_buffer import ReplayBuffer
from low_earth_orbit.util import util, constant
from MA_TD3.misc import misc
import math
import time
import numpy as np
from random import sample

start_time = time.time()

power = util.tolinear(46.40815561244022) + util.tolinear(47.50248870629099)
print(util.todb(power), power)

print(time.time() - start_time)
