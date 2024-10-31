
from MA_TD3.misc.replay_buffer import ReplayBuffer
from random import sample

size = 50
buffer = ReplayBuffer(size)
for i in range(5):
  buffer.add(i + 3)

print(buffer.sample(16))
