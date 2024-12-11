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
    self.name = 'Digital World'
    self.rt_data = misc.load_rt_file(f'ue{len(self.ues)}_rt_result')

    def mean_stdv_of_db(rt_data, key):
      data_array = np.asarray([data[key]
                               for t in rt_data
                               for sat_name in rt_data[t]
                               for b_i in rt_data[t][sat_name]
                               for data in rt_data[t][sat_name][b_i]])
      return np.mean(data_array), np.std(data_array)

    self.path_gain_mean, self.path_gain_stdv = mean_stdv_of_db(self.rt_data, 'path gain (dB)')
    self.path_loss_mean, self.path_loss_stdv = mean_stdv_of_db(self.rt_data, 'path loss (dB)')
    self.hr_mean, self.hr_stdv = mean_stdv_of_db(self.rt_data, 'h_r')
    self.hi_mean, self.hi_stdv = mean_stdv_of_db(self.rt_data, 'h_i')
    self.prev_rt_state = {}

  def get_rt_state(self, sat_name):
    rt_info = self.rt_data[self.step_num][sat_name]
    path_gain_res = np.zeros((self.cell_num, ))
    # path_loss_res = np.zeros((self.cell_num, ))
    h_r_res = np.zeros((self.cell_num, ))
    h_i_res = np.zeros((self.cell_num, ))

    for b_i in range(self.cell_num):
      for data in rt_info[b_i]:
        ue_idx = data['ue']
        self.ue_dict[f'ue{ue_idx}'].servable_add(
          sat_name, b_i, util.todb(data['received power (W)'] * constant.MILLIWATT))

      path_gain_res[b_i] = max([data['path gain (dB)'] for data in rt_info[b_i]])
      # path_loss_res[b_i] = min([data['path loss (dB)'] for data in rt_info[b_i]])
      h_r_res[b_i] = max([abs(data['h_r']) for data in rt_info[b_i]])
      h_i_res[b_i] = max([abs(data['h_r']) for data in rt_info[b_i]])

    norm_pg = util.standardize(path_gain_res, self.path_gain_mean, self.path_gain_stdv)
    # norm_pl = util.standardize(path_loss_res, self.path_loss_mean, self.path_loss_stdv)
    norm_hr = util.standardize(h_r_res, self.hr_mean, self.hr_stdv)
    norm_hi = util.standardize(h_i_res, self.hi_mean, self.hi_stdv)

    return np.concatenate((norm_pg, norm_hr, norm_hi))

  def get_state_info(self, has_action=True) -> Dict[str, List[float]]:
    state_dict = {}
    for ue in self.ues:  # Don't move this!!! I fall for this two times QAQ
      ue.servable_clear()

    for sat_name in self.leo_agents:
      if has_action:
        rt_state = self.get_rt_state(sat_name)
        self.prev_rt_state[sat_name] = rt_state
      else:
        rt_state = self.prev_rt_state[sat_name]

      state_dict[sat_name] = np.float32(np.concatenate((self.get_position_state(sat_name),
                                                        rt_state)))
    return state_dict

  def no_action_step(self):
    ue_sinr = self.constel.cal_transmission_sinr(ues=self.ues,
                                                 interference_beams=self.additional_beam_set)
    ue_throughput = self.constel.cal_throughput(ues=self.ues,
                                                sinr=ue_sinr,
                                                interference_beams=self.additional_beam_set)
    reward = self._cal_reward(ue_throughput=ue_throughput, no_action=True)
    self.record_steps_of_last_ep(ue_sinr=ue_sinr, ue_throughput=ue_throughput)

    self.step_num += 1
    self.total_step_num += 1
    self.constel.update_sat_position()
    done = (self.step_num >= self.max_step)
    truncated = (self.step_num >= self.max_step)
    has_action = (self.step_num % self.action_period == 0)
    obs = self.get_state_info(has_action)

    return (obs, reward, done, truncated, {'has_action': has_action})

  def _cal_overhead(self, agent: Agent) -> float:
    leo2dt_distance = self.dt_server.position.calculate_distance(agent.sat.position)

    realworld_header = self.real_agents[agent.sat_name].sat.beam_training_latency

    digitalworld_header = util.rt_delay(len(self.digital_agents) * len(self.ues),
                                        self.digital_agents[agent.sat_name].comp_freq)

    state_exchange_overhead = (agent.sat.trans_latency(agent.state_dim * constant.FLOAT_SIZE, self.dt_server)
                               + self.dt_server.trans_latency(agent.state_dim * constant.FLOAT_SIZE)
                               + 2 * util.propagation_delay(leo2dt_distance))

    leo_feedback_size = ((len(self.ues) + self.real_agents[agent.sat_name].twin_sharing_param_num / self.args.model_sharing_period)
                         * constant.FLOAT_SIZE)
    leo_feedback_latency = agent.sat.trans_latency(leo_feedback_size, self.dt_server)

    dt_feedback_size = ((self.digital_agents[agent.sat_name].twin_sharing_param_num / self.args.model_sharing_period)
                        * constant.FLOAT_SIZE)

    dt_feedback_latency = self.dt_server.trans_latency(dt_feedback_size)

    feedback_overhead = (leo_feedback_latency
                         + dt_feedback_latency
                         + 2 * util.propagation_delay(leo2dt_distance))

    overhead = (max(realworld_header, digitalworld_header)
                + state_exchange_overhead
                + agent.computation_latency
                + feedback_overhead)

    self.tb_writer.add_scalars(f'{self.name} Overhead/overhead',
                               {agent.name: overhead},
                               self.total_step_num)
    self.tb_writer.add_scalars(f'{self.name} Overhead/realworld header overhead',
                               {agent.name: realworld_header},
                               self.total_step_num)
    self.tb_writer.add_scalars(f'{self.name} Overhead/digitalworld header overhead',
                               {agent.name: digitalworld_header},
                               self.total_step_num)
    self.tb_writer.add_scalars(f'{self.name} Overhead/state exchange overhead overhead',
                               {agent.name: state_exchange_overhead},
                               self.total_step_num)
    self.tb_writer.add_scalars(f'{self.name} Overhead/comp header',
                               {agent.name: agent.computation_latency},
                               self.total_step_num)
    self.tb_writer.add_scalars(f'{self.name} Overhead/feedback overhead',
                               {agent.name: feedback_overhead},
                               self.total_step_num)
    return overhead
