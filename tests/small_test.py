
from MA_TD3.agent import Agent
from MA_TD3.misc.replay_buffer import ReplayBuffer
from low_earth_orbit.util import util, constant
from MA_TD3.misc import misc
import functools
import timeit
import numpy as np
from random import sample


def test_func(a):
  a = np.clip(a, -2, 5)


if __name__ == '__main__':
  a = [i - 3 for i in range(10)]
  print(timeit.timeit(functools.partial(test_func, a), number=round(1e4)))
