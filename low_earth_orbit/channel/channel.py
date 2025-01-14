"""The channel model."""

from typing import Tuple, List
import functools
import math
import random
import numpy as np
import numpy.typing as npt
from scipy.stats import norm, nakagami
from scipy.interpolate import RegularGridInterpolator
from itur.models import itu676
from low_earth_orbit.util.position import Position
from ..util import constant
from ..util.distribution import Rainfall_rv
from .. import util


class Channel():
  """The class of wireless channel"""

  def __init__(self,
               rain_prob: float,
               has_weather: int,
               month: int = 1):
    self.rain_prob = rain_prob
    self.month = month
    self.has_weather = (has_weather == 1)
    if self.has_weather:
      self.map_lon_list = np.linspace(constant.ORIGIN_LONG - constant.MAP_HALF_WIDTH,
                                      constant.ORIGIN_LONG + constant.MAP_HALF_WIDTH,
                                      num=round(2 * constant.MAP_HALF_WIDTH))
      self.map_lat_list = np.linspace(constant.ORIGIN_LATI - constant.MAP_HALF_WIDTH,
                                      constant.ORIGIN_LATI + constant.MAP_HALF_WIDTH,
                                      num=round(2 * constant.MAP_HALF_WIDTH))
      self.load_rainfall_data()
      self.update_weather()

  def free_space(self, distance: float, freq: float) -> float:
    """The free space path loss model.

    Args:
      distance (float): The propagation distance.
      freq (float): The center frequency.

    Returns:
      (float): The path loss in dB
    """
    pl = 2 * util.todb(distance) + 2 * util.todb(freq) + constant.FREE_SPACE_LOSS
    pl = float(pl)
    return pl

  def scintillation_loss(self, elevation_angle):
    """The scintillation loss model.

    Args:
      elevation_angle (float): The elevation angle in radian.

    Returns:
      (float): The scintillation loss in dB
    """
    rounded_epsilon = np.around(elevation_angle / constant.PI_IN_RAD, decimals=-1)
    epsilon_index = (
        rounded_epsilon /
        round(constant.ANGLE_RESOLUTION / constant.PI_IN_RAD)).astype(int)
    scint_loss = constant.SCINTILLATION_TABLE[epsilon_index]
    return scint_loss

  def gas_attenuation(self,
                      fc: float,
                      elevation_angle: float,
                      water_vapor_density: float,
                      temperature: float,
                      atmos_pressure: float):
    """The gas attenuation model.
    The gas attenuation is zenith loss / sin(elevation_angle).
    The longer the distance through the troposphere (small elevation angle),
    the more severe attenuation it suffers

    Args:
      fc (float): The central frequency of the signal
      elevation_angle (float): The elevation angle in radian.

    Returns:
      (float): The gas attenuation in dB
    """
    return self.zenith_attenuation(fc=fc,
                                   water_vapor_density=water_vapor_density,
                                   temperature=temperature,
                                   atmos_pressure=atmos_pressure) / np.sin(elevation_angle)

  @functools.cache
  def zenith_attenuation(self,
                         fc: float,
                         water_vapor_density: float,
                         temperature: float,
                         atmos_pressure: float) -> float:
    """The zenith attenuation model.
    Zenith loss is the loss when the elevation angle is 90 degrees.

    Args:
      fc (float): The central frequency of the signal

    Returns:
      (float): The zenith attenuation in dB
    """
    zenith_att = itu676.gaseous_attenuation_slant_path(
        f=fc / 1e9,
        el=constant.PI / constant.PI_IN_RAD / 2,
        rho=water_vapor_density,
        P=atmos_pressure,
        T=temperature,
        mode='approx')

    return float(zenith_att.value)

  def cal_deterministic_loss(self,
                             ue_pos: Position,
                             distance: float,
                             freq: float,
                             elevation_angle: float,
                             water_vapor_density: float,
                             temperature: float,
                             atmos_pressure: float,
                             ) -> float:
    """Calculate the deterministic part of the loss.

    Args:
      distance (float): The distance between sat and ue.
      freq (float): The center freq.
      elevation_angle (float): The elevation angle pointing from ue to sat

    Returns:
      (float): The total deterministic loss (dB)
    """
    fspl = self.free_space(distance=distance, freq=freq)
    scpl = self.scintillation_loss(elevation_angle)
    gpl = self.gas_attenuation(freq, elevation_angle,
                               water_vapor_density, temperature, atmos_pressure)

    ue_lon = ue_pos.geodetic.longitude
    ue_lat = ue_pos.geodetic.latitude
    if self.has_weather:
      rl = self.itu_rain_attenuation(rain_rate=self.rain_rate_grid((ue_lon, ue_lat)),
                                     L_s=self.rain_height_grid((ue_lon, ue_lat)) / math.sin(elevation_angle),
                                     freq=freq,
                                     elevation_angle=elevation_angle)
    else:
      rl = 0

    return fspl + scpl + gpl + rl

  def cal_stochastic_loss(self,
                          nakagami_m: float,
                          rx_power_ratio: float,
                          los_power_ratio: float,) -> float:
    """Calculate the stochastic part of the loss.

    Args:

    Returns:
      (float): The total stochastic loss (dB)
    """
    sr_loss = self.shadowed_rician_fading(b=rx_power_ratio - los_power_ratio,
                                          m=nakagami_m,
                                          Omega=los_power_ratio)
    return sr_loss

  def shadowed_rician_fading(self, b: float, m: float, Omega: float) -> float:
    """The shadowed rician fading model.
       Reference: A new simple model for land mobile satellite channels first- and second-order statistics

    Args:
      b (float): half power of multipath component
      m (float): Nakagami parameter
      Omega (float): Average power of LOS component

    Returns:
      (float): The shadow fading in dB
    """
    A = np.random.rayleigh(scale=math.sqrt(b))
    Z = nakagami.rvs(nu=m, scale=math.sqrt(Omega))
    alpha = random.random() * 2 * constant.PI
    R = A * np.exp(1j * alpha) + Z
    return abs(R)

  def cal_total_loss(self,
                     ue_pos: Position,
                     distance: float,
                     freq: float,
                     elevation_angle: float,
                     nakagami_m: float,
                     rx_power_ratio: float,
                     los_power_ratio: float,
                     water_vapor_density: float,
                     temperature: float,
                     atmos_pressure: float) -> float:
    """Calculate the total loss for sat and ue.

    Args:
      distance (float): The distance between sat and ue.
      freq (float): The center freq.
      elevation_angle (float): The elevation angle pointing from ue to sat

    Returns:
      (float): The total loss (dB)
    """
    det_loss = self.cal_deterministic_loss(ue_pos=ue_pos,
                                           distance=round(distance,
                                                          constant.CACHED_PRECISION),
                                           freq=round(freq,
                                                      constant.CACHED_PRECISION),
                                           elevation_angle=round(elevation_angle,
                                                                 constant.CACHED_PRECISION),
                                           water_vapor_density=round(water_vapor_density,
                                                                     constant.CACHED_PRECISION),
                                           temperature=round(temperature,
                                                             constant.CACHED_PRECISION),
                                           atmos_pressure=round(atmos_pressure,
                                                                constant.CACHED_PRECISION)
                                           )

    stoch_loss = self.cal_stochastic_loss(nakagami_m, rx_power_ratio, los_power_ratio)

    return det_loss + stoch_loss

  def rician_fading(self, k_db: float) -> float:
    """Rician fading.

      Args:
        k_db (float): The ratio between the power of LOS and NLOS (dB).

      Returns:
        float: The fading gain (dB).
    """
    K = util.tolinear(k_db)
    mu = np.sqrt(K / (K + 1))
    sigma = np.sqrt(1 / (2 * (K + 1)))

    h = (sigma * norm.rvs() + mu) + 1j * (sigma * norm.rvs() + mu)
    fading_gain = np.abs(h)
    return util.todb(fading_gain)

  def rayleigh_fading(self, b: float) -> float:
    """Rayleigh fading.

    Returns:
      float: The fading gain (dB).
    """
    return self.rician_fading(k_db=constant.MIN_NEG_FLOAT)

  def _rain_fall_prob_param(self, lon: float, lat: float) -> Tuple[float, float]:
    temp = self.mean_temp_grid((lon, lat))
    if temp >= 0:
      r = 0.5874 * np.exp(0.0883 * temp)
    else:
      r = 0.5874

    p_0 = self.mean_rainfall_grid((lon, lat)) / (24 * constant.DAY_IN_MONTH[self.month] * r)
    if p_0 > 0.7:
      p_0 = 0.7
      r = 1 / 0.7 * (self.mean_rainfall_grid((lon, lat)) / (24 * constant.DAY_IN_MONTH[self.month]))

    return r, p_0

  def generate_rainfall(self, lon: float, lat: float) -> List[float]:
    """Generate the rainfall of the given area.
    WARNING: This method is just a rough estimation using data from ITU.

    Args:
        lon (float): longitude
        lat (float): latitude

    Returns:
        List[float]: rainfall (mm/hr)
    """
    r, p_0 = self._rain_fall_prob_param(lon=lon, lat=lat)
    rainfall_rv = Rainfall_rv(r=r, p_0=p_0)
    sample = 20
    skew = max(round(random.random() * sample), 1)
    return np.mean(np.partition(rainfall_rv.rvs(size=sample), -skew)[-skew:])

  def rainrate_of_rain_prob(self, lon: float, lat: float, rain_prob: float):
    if rain_prob < 0 or rain_prob > 1:
      raise ValueError('Rainfall probability need to be between [0, 1].')

    rand = random.random()
    if rain_prob >= rand:
      rainrate = 0
      while rainrate == 0:
        rainrate = self.generate_rainfall(lon=lon, lat=lat)
      return rainrate
    else:
      return 0

  def update_weather(self):
    """Update the rain rate map.
    Assume the rainfall probabilty is identical in everywhere.
    """
    if not self.has_weather:
      return
    rain_rate_map = [[self.rainrate_of_rain_prob(lon=lon,
                                                 lat=lat,
                                                 rain_prob=self.rain_prob) for lat in self.map_lat_list]
                     for lon in self.map_lon_list]
    self.rain_rate_grid = RegularGridInterpolator((self.map_lon_list, self.map_lat_list), rain_rate_map)

  def modified_rain_attenuation(self, rain_rate: float, L_s: float, height_diff: float, freq: float, elevation_angle: float, polarization_angle: float = 0) -> float:
    """Calculate the rain attenuation.
    Reference paper: Rain Attenuation Prediction Model for Satellite Communications in Tropical Regions.

    Args:
        rain_rate (float): rain rate (mm/h)
        L_s (float): slant path distance (m)
        height_diff: height from the rain top to the ground user.
        freq (float): central frequency
        elevation_angle (float): elevation angle
        polarization_angle (float, optional): polarization angle. Defaults to 0.

    Returns:
        float: rain attenuation (dB)
    """
    k_h, alpha_h, k_v, alpha_v = constant.COEFFICIENT_TABLE_FOR_RAIN_ATTENUATION[round(freq / 1e9)]
    k = (k_h + k_v + (k_h - k_v) * (math.cos(elevation_angle) ** 2) * math.cos(2 * polarization_angle)) / 2
    alpha = (k_h * alpha_h + k_v * alpha_v + (k_h * alpha_h - k_v * alpha_v) *
             (math.cos(elevation_angle) ** 2) * math.cos(2 * polarization_angle)) / (2 * k)
    a_1, a_2, a_3, a_4 = (0.3979, 0.0021, 0.0185, 0.2337)
    r = 1 / (a_1 / math.sin(elevation_angle) + a_2 * rain_rate * height_diff / constant.KM - a_3 * (freq / 1e9) + a_4)
    print("coeffiecnt", k, alpha, r)
    A_p = k * (rain_rate ** alpha) * L_s / constant.KM * r
    return A_p

  def itu_rain_attenuation(self, rain_rate: float, L_s: float, freq: float, elevation_angle: float, polarization_angle: float = 0) -> float:
    """Calculate the rain attenuation suggested by ITU-R P.838

    Args:
        rain_rate (float): rain rate (mm/h)
        L_s (float): slant path distance (m)
        freq (float): central frequency
        elevation_angle (float): elevation angle
        polarization_angle (float, optional): polarization angle. Defaults to 0.

    Returns:
        float: rain attenuation (dB)
    """
    if rain_rate == 0:
      return 0
    k_h, alpha_h, k_v, alpha_v = constant.COEFFICIENT_TABLE_FOR_RAIN_ATTENUATION[round(freq / 1e9)]
    k = (k_h + k_v + (k_h - k_v) * (math.cos(elevation_angle) ** 2) * math.cos(2 * polarization_angle)) / 2
    alpha = (k_h * alpha_h + k_v * alpha_v + (k_h * alpha_h - k_v * alpha_v) *
             (math.cos(elevation_angle) ** 2) * math.cos(2 * polarization_angle)) / (2 * k)
    return k * (rain_rate ** alpha) * L_s / constant.KM

  def load_rainfall_data(self):
    mean_surface_temp = []
    mean_total_rainfall = []
    rain_rate_exceed_001 = []
    rain_height = []

    with open(f'low_earth_orbit/util/rainfall_data/mean surface temperature/LAT_T_LIST.TXT', mode='r', newline='') as f:
      T_lat_list = [float(data) for data in f.read().split(' ')]
    with open(f'low_earth_orbit/util/rainfall_data/mean surface temperature/LON_T_LIST.TXT', mode='r', newline='') as f:
      T_lon_list = [float(data) for data in f.read().split(' ')]
    with open(f'low_earth_orbit/util/rainfall_data/mean surface temperature/T_Month{self.month:0>2}.TXT', mode='r', newline='') as f:
      for line in f.read().splitlines():
        mean_surface_temp.append([float(data) + constant.KELVIN_TO_CELCIUS for data in line.split(' ')])
    mean_surface_temp = list(map(list, zip(*mean_surface_temp)))  # transpose 2d list

    with open(f'low_earth_orbit/util/rainfall_data/mean total rainfall/LAT_MT_LIST.TXT', mode='r', newline='') as f:
      MT_lat_list = [float(data) for data in f.read().split(' ')]
    with open(f'low_earth_orbit/util/rainfall_data/mean total rainfall/LON_MT_LIST.TXT', mode='r', newline='') as f:
      MT_lon_list = [float(data) for data in f.read().split(' ')]
    with open(f'low_earth_orbit/util/rainfall_data/mean total rainfall/MT_Month{self.month:0>2}.TXT', mode='r', newline='') as f:
      for line in f.read().splitlines():
        mean_total_rainfall.append([float(data) for data in line.split(' ')])
    mean_total_rainfall = list(map(list, zip(*mean_total_rainfall)))

    with open(f'low_earth_orbit/util/rainfall_data/rainfall rate exceeded 0.01 percent/LAT_R001_LIST.TXT', mode='r', newline='') as f:
      R001_lat_list = [float(data) for data in f.read().split(' ')]
    with open(f'low_earth_orbit/util/rainfall_data/rainfall rate exceeded 0.01 percent/LON_R001_LIST.TXT', mode='r', newline='') as f:
      R001_lon_list = [float(data) for data in f.read().split(' ')]
    with open(f'low_earth_orbit/util/rainfall_data/rainfall rate exceeded 0.01 percent/R001.TXT', mode='r', newline='') as f:
      for line in f.read().splitlines():
        rain_rate_exceed_001.append([float(data) for data in line.split(' ')])
    rain_rate_exceed_001 = list(map(list, zip(*rain_rate_exceed_001)))

    with open(f'low_earth_orbit/util/rainfall_data/rain height/LAT_h0_LIST.TXT', mode='r', newline='') as f:
      rain_height_lat_list = [float(data) for data in f.read().split(' ')]
    with open(f'low_earth_orbit/util/rainfall_data/rain height/LON_h0_LIST.TXT', mode='r', newline='') as f:
      # print(f.read().split(' '))
      rain_height_lon_list = [float(data) for data in f.read().split(' ')]
    with open(f'low_earth_orbit/util/rainfall_data/rain height/h0.TXT', mode='r', newline='') as f:
      for line in f.read().splitlines():
        rain_height.append([float(data) * constant.KM + constant.ZERO_DEGREE_ISOTHERM_HEIGHT
                           for data in line.split(' ')])
    rain_height = list(map(list, zip(*rain_height)))

    self.mean_temp_grid = RegularGridInterpolator((T_lon_list, T_lat_list), mean_surface_temp)
    self.mean_rainfall_grid = RegularGridInterpolator((MT_lon_list, MT_lat_list), mean_total_rainfall)
    self.rain_rate_001_grid = RegularGridInterpolator((R001_lon_list, R001_lat_list), rain_rate_exceed_001)
    self.rain_height_grid = RegularGridInterpolator((rain_height_lon_list, rain_height_lat_list), rain_height)
