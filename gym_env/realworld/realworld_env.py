"""leosat_env.py"""
from typing import List, Set, Dict
import numpy as np
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
               tb_writer,
               agent_dict: Dict[str, Agent],
               real_agents: Dict[str, Agent],
               digital_agents: Dict[str, Agent],
               agent_names: List[str]):
    super().__init__(ax=ax,
                     args=args,
                     tb_writer=tb_writer,
                     agent_dict=agent_dict,
                     real_agents=real_agents,
                     digital_agents=digital_agents,
                     agent_names=agent_names)
    self.name = 'Real World'
    self.twin_online = False

    self.max_sinr = 20
    self.min_sinr = -self.max_sinr

  def get_sinr_diff_state(self, cell_sinr, beam_power, sat_name) -> npt.NDArray[np.float32]:
    sinr_diff = cell_sinr[sat_name] - self.prev_cell_sinr[sat_name]
    power_diff = beam_power[sat_name] - self.prev_beam_power[sat_name]

    estimated_channel_diff = sinr_diff - power_diff
    closed_beam_bool = np.logical_or(
        cell_sinr[sat_name] == 0, self.prev_cell_sinr[sat_name] == 0)
    estimated_channel_diff[closed_beam_bool] = self.min_sinr  # Set the diff of closed beams to -inf

    sinr_diff_state = np.clip(estimated_channel_diff,
                              self.min_sinr,
                              self.max_sinr) / self.max_sinr

    self.prev_cell_sinr[sat_name] = cell_sinr[sat_name]
    self.prev_beam_power[sat_name] = beam_power[sat_name]

    return sinr_diff_state

  def get_beam_training_state(self) -> npt.NDArray[np.float32]:
    agent = self.leo_agents[self.agent_names[0]]
    state = np.zeros(len(agent.observation_space.low[agent.beam_info_slice]))
    state_dict = {}

    # Beam Training
    ues_sinr = self.constel.scan_ues(ues=self.ues, sat_name_list=self.agent_names)

    for sat_name in self.agent_names:
      # Sinr feedback to state
      for i in range(len(state)):
        state[i] = max([sinr_list[i] for sinr_list in ues_sinr[sat_name].values()])

      state_dict[sat_name] = np.clip(state,
                                     self.min_sinr,
                                     self.max_sinr) / self.max_sinr

    return state_dict

  def get_state_info(self, init=False) -> Dict[str, List[float]]:
    state_dict = {}

    bt_state_dict = self.get_beam_training_state()

    for sat_name in self.leo_agents:
      state_dict[sat_name] = np.float32(np.concatenate((self.get_position_state(sat_name),
                                                        bt_state_dict[sat_name])))

    return state_dict

  def _cal_overhead(self, agent: Agent) -> float:
    leo2dt_distance = self.dt_server.position.calculate_distance(agent.sat.position)

    realworld_header = agent.sat.beam_training_latency
    if self.twin_online:
      digitalworld_header = (util.rt_delay(len(self.leo_agents) * len(self.ues))
                             + self.dt_server.trans_latency(agent.state_dim * constant.INT_SIZE)
                             + util.propagation_delay(leo2dt_distance))
    else:
      digitalworld_header = 0

    feedback_overhead = (agent.sat.trans_latency(len(self.ues) * constant.FLOAT_SIZE, self.dt_server)
                         + util.propagation_delay(leo2dt_distance))

    overhead = (max(realworld_header, digitalworld_header)
                + agent.computation_latency + feedback_overhead)

    self.tb_writer.add_scalars(f'{self.name} Env Param/overhead',
                               {agent.name: overhead},
                               self.step_num + (self.reset_count - 1) * self.max_step)
    self.tb_writer.add_scalars(f'{self.name} Env Param/realworld_header overhead',
                               {agent.name: realworld_header},
                               self.step_num + (self.reset_count - 1) * self.max_step)
    self.tb_writer.add_scalars(f'{self.name} Env Param/digitalworld_header overhead',
                               {agent.name: digitalworld_header},
                               self.step_num + (self.reset_count - 1) * self.max_step)
    self.tb_writer.add_scalars(f'{self.name} Env Param/comp header',
                               {agent.name: agent.computation_latency},
                               self.step_num + (self.reset_count - 1) * self.max_step)
    self.tb_writer.add_scalars(f'{self.name} Env Param/feedback_overhead',
                               {agent.name: feedback_overhead},
                               self.step_num + (self.reset_count - 1) * self.max_step)
    return overhead
