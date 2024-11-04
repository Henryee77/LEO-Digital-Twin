from low_earth_orbit.channel.channel import Channel
from low_earth_orbit.util import constant

channel = Channel(month=1)

print(channel.rain_rate_001_grid((constant.ORIGIN_LONG, constant.ORIGIN_LATI - 1.5)))
print(channel.mean_rainfall_grid((constant.ORIGIN_LONG, constant.ORIGIN_LATI - 1.5)))
