"""leosat_env.py"""
from typing import List, Set, Dict
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from gym_env.leosat.leosat_env import LEOSatEnv
from low_earth_orbit.util import util
from low_earth_orbit.util import constant
from MA_TD3.agent.agent import Agent


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

  def _take_action(self, action, sat_name):
    train_set = self.action_to_trainset(action[self.leo_agents[sat_name].beam_slice])
    # print(train_set)

    self.constel.assign_training_set(sat_name=sat_name,
                                     train_set=train_set)

    beamwidth_dict = self.action_to_beamwidth_dict(action[self.leo_agents[sat_name].beamwidth_slice], sat_name=sat_name)
    for beam_idx, beamwidth in beamwidth_dict.items():
      self.leo_agents[sat_name].sat.set_beamwidth(beam_idx=beam_idx, beamwidth=beamwidth)
      # print(f'{beam_idx}, {beamwidth / constant.PI_IN_RAD}')

    self.leo_agents[sat_name].sat.scan_beam()

  def action_to_beamwidth_dict(self, beamwidth_action: npt.NDArray[np.float64], sat_name) -> Dict[int, float]:
    res = {}
    agent = self.leo_agents[sat_name]
    for i, beamwidth in enumerate(beamwidth_action):
      res[i] = util.rescale_value(beamwidth,
                                  agent.beamwidth_action_low[i],
                                  agent.beamwidth_action_high[i],
                                  agent.min_beamwidth,
                                  agent.max_beamwidth)

    return res

  def action_to_power_dict(self, power_action: npt.NDArray[np.float64], sat_name) -> Dict[int, float]:
    """Map the action output to the beam tx power.
    Args:
        power_action (npt.NDArray[np.float64]): Action of the power
                                                of all beams and total power.
    Returns:
        Dict[int, float]: Dict of the power of each turned on beam.
    """
    # print(f'action: {power_action}')
    online_beam = self.leo_agents[sat_name].sat.cell_topo.online_beam_set
    power_dict = {}
    for beam_idx in range(self.cell_num):
      if beam_idx in online_beam:
        power_dict[beam_idx] = abs(power_action[beam_idx])
      else:
        power_dict[beam_idx] = 0
    agent = self.leo_agents[sat_name]
    total_power = util.tolinear(util.rescale_value(power_action[-1],
                                                   agent.total_power_low[0],
                                                   agent.total_power_high[0],
                                                   agent.min_power,
                                                   agent.max_power))

    multiplier = (total_power
                  / (np.sum(np.fromiter(power_dict.values(), dtype=float)) + constant.MIN_POSITIVE_FLOAT))
    for beam_idx in online_beam:
      power_dict[beam_idx] *= multiplier

    total_power = np.sum(np.fromiter(power_dict.values(), dtype=float))
    # print(f'total power: {util.todb(total_power)}')
    for beam_idx in power_dict:
      power_dict[beam_idx] = util.todb(util.truncate(power_dict[beam_idx]))

    return power_dict
