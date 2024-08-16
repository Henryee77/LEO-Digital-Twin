"""The constant used in project."""

import sys
import math

import numpy as np
import itur

# Math
PI = math.pi
PI_IN_RAD = math.pi / 180
KM = 1000

# Nature
R_EARTH = 6371 * KM  # 6371 km
ANGULAR_SPEED_EARTH = 7.292115856e-5
LIGHT_SPEED = 299792458
THERMAL_NOISE_POWER = -174
STAND_GRAVIT_PARA = 3.986004418e14

# Simulation Configuration
TIMESLOT = 2  # training period (sec)

# Beam-Training Parameteres
SINR_THRESHOLD = 10  # (dB)
TRAINING_WINDOW_SIZE = 4
BEAM_TRAINING_ACCURACY_THRESHOLD = 0.75
T_BEAM = 0.15  # (s)
T_FB = 0.01
T_ACK = 0.01
A3_INTER_SAT_OFFSET = 6
A3_INTRA_SAT_OFFSET = 3
A3_HYSTERESIS = 0
A3_RESET = 0
A3_ONE_TIMESLOT = 1
TIME_TO_TRIGGER = 2  # 2 timeslot
OFFLINE_TIME_TO_TRIGGER = 0  # 0 timeslot

# Ground User
DEFAULT_RX_GAIN = 40  # (dB)
EPSILON_SERVING_THRESHOLD = 30 * PI_IN_RAD

# Cell
DEFAULT_CELL_RADIUS = 100 * KM  # 100km
DEFAULT_CELL_LAYER = 3
DEFAULT_SHAPE = 6
CELL_GRID_RESOLUTION = 25
TOPOLOGY_ANGLE = np.linspace(0, 2 * np.pi, DEFAULT_SHAPE + 1)
DRAW_THETA = np.linspace(0, 2 * np.pi, CELL_GRID_RESOLUTION)
MAIN_LOBE_RANGE = 3.3  # The approximation to get the range of main lobe

# Beam
DEFAULT_CENTRAL_FREQUENCY = 20e9  # 20 GHz
DEFAULT_BANDWIDTH = 100e6  # 100 MHz
DEFAULT_BEAMWIDTH_3DB = 3.3 * PI_IN_RAD  # 3.3 deg
DEFAULT_BEAM_SWEEPING_ALG = 1

# Channel
FREE_SPACE_LOSS = -147.55  # 20 * log10(4 * pi / c)
SF_MODEL = 0  # 0 = dense urban, 1 = urban, 2 = suburban
ANGLE_RESOLUTION = 10 * PI_IN_RAD
SF_STDV_LIST: list[list[float]] = [
    [2.9, 2.9, 2.4, 2.7, 2.4, 2.4, 2.7, 2.6, 2.8, 0.6],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [1.9, 1.9, 1.6, 1.9, 2.3, 2.7, 3.1, 3.0, 3.6, 0.4],
]
SCINTILLATION_TABLE = np.array(
    [1.08, 1.08, 0.48, 0.30, 0.22, 0.17, 0.13, 0.12, 0.12, 0.12])
GROUND_WATER_VAP_DENSITY = 7.5 * itur.u.g / itur.u.m**3
GROUND_ATMOS_PRESSURE = 1013.25 * itur.u.hPa
GROUND_TEMPERATURE = 25 * itur.u.deg_C
SCATTER_COMPONENT_HALF_POWER = 0.126
NAKAGAMI_PARAMETER = 10
LOS_COMPONENT_POWER = 0.835
COEFFICIENT_TABLE_FOR_RAIN_ATTENUATION = {20: (0.09164, 1.0568, 0.09611, 0.9847)}
CACHED_PRECISION = 2

# Antenna
DEFAULT_APERTURE_SIZE = 0.15  # (m)
DEFAULT_ANTENNA_EFFICIENCY = 0.8
ANT_GAIN_COEFF = 2.07123

# because one of a denominator in the antenna gain fomula is raised to the power 3
MIN_POSITIVE_FLOAT = float(sys.float_info.min**(1. / 4))
MIN_NEG_FLOAT = -1 * sys.float_info.max

# Constellation
ORIGIN_LONG = 121
ORIGIN_LATI = 24

# Sat
MAX_POWER = 50  # dBm

# Plot
SAT_MARKER_SIZE = 100
UE_MARKER_SIZE = 50
DEFAULT_TOPO_CENTER_COLOR = 'b'
DEFAULT_CELL_NORM_COLOR = 'k'
DEFAULT_CELL_SCAN_COLOR = 'r'
DEFAULT_CELL_SERV_COLOR = 'g'
CELL_ALPHA = 0.5

# Beam training
DEFAULT_TRAINING_WINDOW = 4
