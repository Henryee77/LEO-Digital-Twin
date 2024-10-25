"""leosat_env.py"""
from typing import List, Set, Dict
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from gym_env.leosat.leosat_env import LEOSatEnv
from low_earth_orbit.util import util
from low_earth_orbit.util import constant
from MA_TD3.agent.agent import Agent
from MA_TD3.misc import misc


class DigitalWorldEnv(LEOSatEnv):
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
    self.name = 'Digital World'
    self.rt_data = misc.load_rt_file()

    path_loss_array = np.asarray([data['path loss (dB)']
                                  for t in self.rt_data
                                  for sat_name in self.rt_data[t]
                                  for b_i in self.rt_data[t][sat_name]
                                  for data in self.rt_data[t][sat_name][b_i]])

    self.path_loss_mean = np.mean(path_loss_array)
    self.path_loss_stdv = np.std(path_loss_array)

  def get_rt_state(self, sat_name):
    rt_info = self.rt_data[self.step_num][sat_name]
    res = np.zeros(len(self.cell_num, ))
    for b_i in range(self.cell_num):
      res[b_i] = min([data['path loss (dB)'] for data in rt_info[b_i]])

    return util.standardize(res, self.path_loss_mean, self.path_loss_stdv)

  def get_state_info(self, init=False) -> Dict[str, List[float]]:
    state_dict = {}

    for sat_name in self.leo_agents:
      state_dict[sat_name] = np.float32(np.concatenate((self.get_position_state(sat_name),
                                                        self.get_rt_state(sat_name))))
    return state_dict
