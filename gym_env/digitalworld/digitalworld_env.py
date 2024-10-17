"""leosat_env.py"""
from typing import List, Set, Dict

from matplotlib.axes import Axes
from gym_env.leosat.leosat_env import LEOSatEnv


class DigitalWorldEnv(LEOSatEnv):
  """The LEO Env class."""

  def __init__(self, ax: Axes, args, agent_dict, agent_names: List[str]):
    super().__init__(ax=ax,
                     args=args,
                     agent_dict=agent_dict,
                     agent_names=agent_names)
    self.name = 'Digital World'
