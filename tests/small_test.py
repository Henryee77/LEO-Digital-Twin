
from MA_TD3.misc.replay_buffer import ReplayBuffer
from low_earth_orbit.util import util
import math
import time
from random import sample

start_time = time.time()

lst = [('1', 10), ('1', 3)]
a = set(x for x in lst)
print(a)
tup = ('1', 10)
print(a.difference(tup))

print(time.time() - start_time)
