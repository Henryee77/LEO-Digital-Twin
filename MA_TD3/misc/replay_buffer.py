"""Replay_buffer.py"""
import copy
import numpy as np
from random import sample


class ReplayBuffer(object):
  def __init__(self, max_size):
    self.buffer = [None] * max_size
    self.max_size = max_size
    self.index = 0
    self.size = 0

  def __len__(self):
    return self.size

  def clear(self):
    self.buffer.clear()
    assert len(self.buffer) == 0

  def sync(self, memory):
    self.clear()
    assert type(memory) is type(self.buffer)
    self.buffer = copy.deepcopy(memory)

    assert len(memory) == len(self.buffer)

  def add(self, data):
    # Expect tuples of (state, next_state, action, reward, done)
    self.buffer[self.index] = data
    self.size = min(self.size + 1, self.max_size)
    self.index = (self.index + 1) % self.max_size

  def sample(self, batch_size):
    indices = sample(range(self.size), batch_size)
    sample_data = [self.buffer[index] for index in indices]
    state, next_state, action, reward, done = zip(*sample_data)
    return (np.array(state),
            np.array(next_state),
            np.array(action),
            np.array(reward).reshape(-1, 1),
            np.array(done).reshape(-1, 1))
