"""leosat_env.py"""
from typing import List, Set, Dict
import numpy.typing as npt
import matplotlib.pyplot as plt

from gym_env.leosat.leosat_env import LEOSatEnv
from low_earth_orbit.ground_user.user import User
from low_earth_orbit.util.position import Position, Geodetic
from low_earth_orbit.util import util, constant
from MA_TD3.agent.agent import Agent


class RealWorldEnv(LEOSatEnv):
  """The LEO Env class."""

  def __init__(self,
               ax: plt.Axes,
               args,
               agent_dict: Dict[str, Agent],
               real_agents: Dict[str, Agent],
               digital_agents: Dict[str, Agent],
               agent_names: List[str]):
    super().__init__(ax=ax,
                     args=args,
                     agent_dict=agent_dict,
                     real_agents=real_agents,
                     digital_agents=digital_agents,
                     agent_names=agent_names)
    self.name = 'Real World'

  def step(self, action_n: Dict[str, npt.NDArray]):
    if self.step_num % self.train_per_move == 0:
      self.constel.update_sat_position()
