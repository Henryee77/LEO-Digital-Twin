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

    self.max_sinr = 20
    self.min_sinr = -self.max_sinr
    self.prev_bt_state = {}

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
    ues_sinr = self.constel.scan_ues(ues=self.ues,
                                     sat_name_list=self.agent_names,
                                     scan_mode=self.args.beam_sweeping_mode)

    for sat_name in self.agent_names:
      # Sinr feedback to state
      for i in range(len(state)):
        state[i] = max([sinr_list[i] for sinr_list in ues_sinr[sat_name].values()])

      state_dict[sat_name] = np.clip(state,
                                     self.min_sinr,
                                     self.max_sinr) / self.max_sinr

    return state_dict

  def get_state_info(self, has_action=True) -> Dict[str, List[float]]:
    state_dict = {}

    if has_action:
      bt_state_dict = self.get_beam_training_state()
      self.prev_bt_state = bt_state_dict
    else:
      bt_state_dict = self.prev_bt_state

    for sat_name in self.leo_agents:
      state_dict[sat_name] = np.float32(np.concatenate((self.get_position_state(sat_name),
                                                        bt_state_dict[sat_name])))

    return state_dict

  def no_action_step(self):

    ue_sinr = self.constel.cal_transmission_sinr(ues=self.ues,
                                                 interference_beams=self.additional_beam_set)
    ue_throughput = self.constel.cal_throughput(ues=self.ues,
                                                sinr=ue_sinr,
                                                interference_beams=self.additional_beam_set)
    reward = self._cal_reward(ue_throughput=ue_throughput, no_action=True)

    self.record_sinr_thpt(ue_sinr=ue_sinr, ue_throughput=ue_throughput)

    self.step_num += 1
    self.constel.update_sat_position()
    done = (self.step_num >= self.max_step)
    truncated = (self.step_num >= self.max_step)
    has_action = (self.step_num % self.action_period == 0)
    obs = self.get_state_info(has_action)
    return (obs, reward, done, truncated, {'has_action': has_action})

  def _cal_overhead(self, agent: Agent) -> float:
    leo2dt_distance = self.dt_server.position.calculate_distance(agent.sat.position)

    if self.twin_online:
      realworld_header = agent.sat.beam_training_latency
      digitalworld_header = (util.rt_delay(len(self.leo_agents) * len(self.ues), self.digital_agents[agent.sat_name].comp_freq)
                             + self.dt_server.trans_latency(agent.state_dim * constant.INT_SIZE)
                             + util.propagation_delay(leo2dt_distance))
      state_exchange_overhead = (agent.sat.trans_latency(agent.state_dim * constant.FLOAT_SIZE, self.dt_server)
                                 + self.dt_server.trans_latency(agent.state_dim * constant.FLOAT_SIZE)
                                 + 2 * util.propagation_delay(leo2dt_distance))
      leo_feedback_size = ((len(self.ues) + self.real_agents[agent.sat_name].twin_sharing_param_num / self.args.twin_sharing_period)
                           * constant.FLOAT_SIZE)
      leo_feedback_latency = agent.sat.trans_latency(leo_feedback_size, self.dt_server)

      dt_feedback_size = ((self.digital_agents[agent.sat_name].twin_sharing_param_num / self.args.twin_sharing_period)
                          * constant.FLOAT_SIZE)

      dt_feedback_latency = self.dt_server.trans_latency(dt_feedback_size)

      feedback_overhead = (leo_feedback_latency
                           + dt_feedback_latency
                           + 2 * util.propagation_delay(leo2dt_distance))
    else:
      realworld_header = agent.sat.beam_training_latency
      digitalworld_header = 0
      state_exchange_overhead = 0
      feedback_overhead = 0

    overhead = (max(realworld_header, digitalworld_header)
                + state_exchange_overhead
                + agent.computation_latency
                + feedback_overhead)

    if self.last_episode:
      self.tb_writer.add_scalars(f'{self.name} Overhead/overhead',
                                 {agent.name: overhead},
                                 self.step_num + (self.reset_count - 1) * self.max_step)
      self.tb_writer.add_scalars(f'{self.name} Overhead/realworld_header overhead',
                                 {agent.name: realworld_header},
                                 self.step_num + (self.reset_count - 1) * self.max_step)
      self.tb_writer.add_scalars(f'{self.name} Overhead/digitalworld_header overhead',
                                 {agent.name: digitalworld_header},
                                 self.step_num + (self.reset_count - 1) * self.max_step)
      self.tb_writer.add_scalars(f'{self.name} Overhead/comp header',
                                 {agent.name: agent.computation_latency},
                                 self.step_num + (self.reset_count - 1) * self.max_step)
      self.tb_writer.add_scalars(f'{self.name} Overhead/feedback_overhead',
                                 {agent.name: feedback_overhead},
                                 self.step_num + (self.reset_count - 1) * self.max_step)
    return overhead
