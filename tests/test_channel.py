"""Test file for channel.py"""
import numpy as np
import matplotlib.pyplot as plt

from low_earth_orbit.channel.channel import Channel
from low_earth_orbit.antenna.antenna import Antenna
from low_earth_orbit.satellite.satellite import Satellite
from low_earth_orbit.ground_user.user import User
from low_earth_orbit.util.position import Position, Geodetic
from low_earth_orbit.util import constant
from low_earth_orbit.util.distribution import Rayleigh, Nakagami

wireless_channel = Channel()
b = 0.126
m = 10
Omega = 0.835

iter = 10000

'''
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
plt.show()'''

sat = Satellite(0, 0, 0, Position(geodetic=Geodetic(120, 25, 500 * constant.KM + constant.R_EARTH)),
                angle_speed=0, cell_topo=None, antenna=Antenna(), channel=wireless_channel)
ue = User("0", position=Position(geodetic=Geodetic(121, 25.5, 200 + constant.R_EARTH)))
distance = sat.position.calculate_distance(ue.position)
elevation_angle = sat.position.cal_elevation_angle(ue.position) / constant.PI_IN_RAD
rain_height = 5 * constant.KM + constant.R_EARTH
# print(sat.position.cartesian, ue.position.cartesian)
L_s = (rain_height - constant.R_EARTH) / np.sin(elevation_angle)
print(distance, elevation_angle, L_s, rain_height - ue.position.geodetic.height)


channel_gain = wireless_channel.itu_rain_attenuation(
    rain_rate=0.3, L_s=L_s / constant.KM, height_diff=rain_height - ue.position.geodetic.height, freq=sat.antenna.central_frequency, elevation_angle=elevation_angle)
print(channel_gain)
