"""leosat_env.py"""
from typing import List, Set, Dict

from matplotlib.axes import Axes
from gym_env.leosat.leosat_env import LEOSatEnv


class RealWorldEnv(LEOSatEnv):
  """The LEO Env class."""

  def __init__(self, ax: Axes, args, agent_dict, agent_names: List[str]):
    super().__init__(ax, args, agent_dict, agent_names)

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
