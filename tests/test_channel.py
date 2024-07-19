"""Test file for channel.py"""
import numpy as np
import matplotlib.pyplot as plt

from low_earth_orbit.channel.channel import Channel
from low_earth_orbit.util.distribution import Rayleigh, Nakagami

wireless_channel = Channel()
b = 0.126
m = 10
Omega = 0.835

iter = 10000

channel_gain = np.zeros(iter)
for i in range(iter):
  channel_gain[i] = wireless_channel.shadowed_rician_fading(b=b, m=m, Omega=Omega)
  # print(channel_gain[i])

plt.hist(channel_gain, bins=100)
print(np.mean(channel_gain), np.var(channel_gain))
plt.show()


ray = Rayleigh()
rayleigh_dist = ray.rvs(scale=np.sqrt(b), size=iter)
plt.hist(rayleigh_dist, bins=100)
print(np.mean(rayleigh_dist), np.var(rayleigh_dist))
plt.show()

nakagami = Nakagami()
nakagami_dist = nakagami.rvs(nu=m, scale=np.sqrt(Omega), size=iter)
plt.hist(nakagami_dist, bins=100)
print(np.mean(nakagami_dist), np.var(nakagami_dist))
plt.show()
