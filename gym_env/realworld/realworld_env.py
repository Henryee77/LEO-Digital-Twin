"""leosat_env.py"""
from typing import List, Set, Dict

import numpy.typing as npt
from matplotlib.axes import Axes
from gym_env.leosat.leosat_env import LEOSatEnv
from low_earth_orbit.ground_user.user import User
from low_earth_orbit.util.position import Position, Geodetic
from low_earth_orbit.util import constant


class RealWorldEnv(LEOSatEnv):
  """The LEO Env class."""

  def __init__(self, ax: Axes, args, agent_dict, agent_names: List[str]):
    super().__init__(ax, args, agent_dict, agent_names)
    self.name = 'Real World'
    self.dt_server = User('DT server', position=Position(geodetic=Geodetic(longitude=constant.ORIGIN_LONG,
                                                                           latitude=constant.ORIGIN_LATI,
                                                                           height=constant.R_EARTH)))

  def step(self, action_n: Dict[str, npt.NDArray]):
    if self.step_num % self.train_per_move == 0:
      self.constel.update_sat_position()

  def _take_action(self, action, sat_name):
    train_set = self.action_to_trainset(action[self.beam_slice])
    # print(train_set)

    self.constel.assign_training_set(sat_name=sat_name,
                                     train_set=train_set)

    beamwidth_dict = self.action_to_beamwidth_dict(action[self.beamwidth_slice])
    for beam_idx, beamwidth in beamwidth_dict.items():
      self.main_sats[sat_name].set_beamwidth(beam_idx=beam_idx, beamwidth=beamwidth)
      # print(f'{beam_idx}, {beamwidth / constant.PI_IN_RAD}')

    self.main_sats[sat_name].scan_beam()
