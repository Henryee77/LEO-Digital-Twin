
from MA_TD3.misc.replay_buffer import ReplayBuffer
from low_earth_orbit.util import util
import math
import time
from random import sample

start_time = time.time()
A = [1, 2, 3]
count = {}
for _ in range(1_000_000):
  for a in A:
    count[a] = 2
    count[a] = min(a, count[a] - 10)

print(time.time() - start_time)
