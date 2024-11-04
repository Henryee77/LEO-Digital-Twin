from low_earth_orbit.channel.channel import Channel
from low_earth_orbit.util import constant
import time

channel = Channel(month=1)

lon = constant.ORIGIN_LONG
lat = constant.ORIGIN_LATI - 1.5

# print(channel.rain_rate_001_grid((lon, lat)))
# print(channel.mean_rainfall_grid((lon, lat)))

start_time = time.time()

a = [max(channel.generate_rainfall(lon, lat)) for i in range(80)]
print(a)
print(time.time() - start_time)
