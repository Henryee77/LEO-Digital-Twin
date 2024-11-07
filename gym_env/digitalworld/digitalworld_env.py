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
    self.rt_data = misc.load_rt_file(f'rt_result_ue{len(self.ues)}')

    path_loss_array = np.asarray([data['path loss (dB)']
                                  for t in self.rt_data
                                  for sat_name in self.rt_data[t]
                                  for b_i in self.rt_data[t][sat_name]
                                  for data in self.rt_data[t][sat_name][b_i]])

    self.path_loss_mean = np.mean(path_loss_array)
    self.path_loss_stdv = np.std(path_loss_array)

  def get_rt_state(self, sat_name):
    rt_info = self.rt_data[self.step_num][sat_name]
    res = np.zeros((self.cell_num, ))
    for ue in self.ues:
      ue.servable_clear()

    for b_i in range(self.cell_num):
      for data in rt_info[b_i]:
        ue_idx = data['ue']
        self.ue_dict[f'ue{ue_idx}'].servable_add(
          sat_name, b_i, util.todb(data['received power (W)'] * constant.MILLIWATT))

      res[b_i] = min([data['path loss (dB)'] for data in rt_info[b_i]])

    return util.standardize(res, self.path_loss_mean, self.path_loss_stdv)

  def get_state_info(self, ray_tracing=True) -> Dict[str, List[float]]:
    state_dict = {}

    for sat_name in self.leo_agents:
      if ray_tracing:
        rt_state = self.get_rt_state(sat_name)
        self.prev_rt_state = rt_state
      else:
        rt_state = self.prev_rt_state

      state_dict[sat_name] = np.float32(np.concatenate((self.get_position_state(sat_name),
                                                        rt_state)))
    return state_dict

  def no_action_step(self):
    self.constel.update_sat_position()
    self.step_num += 1
    ue_sinr = self.constel.cal_transmission_sinr(ues=self.ues,
                                                 interference_beams=self.additional_beam_set)
    ue_throughput = self.constel.cal_throughput(ues=self.ues,
                                                sinr=ue_sinr,
                                                interference_beams=self.additional_beam_set)
    self._cal_reward(ue_throughput=ue_throughput)

    done = (self.step_num >= self.max_step)
    truncated = (self.step_num >= self.max_step)
    obs = self.get_state_info(ray_tracing=False)
    return (obs, self.reward, done, truncated, {})

  def _cal_overhead(self, agent: Agent) -> float:
    leo2dt_distance = self.dt_server.position.calculate_distance(agent.sat.position)

    realworld_header = agent.sat.beam_training_latency
    digitalworld_header = (util.rt_delay(len(self.ues))
                           + self.dt_server.trans_latency(agent.state_dim * constant.INT_SIZE)
                           + util.propagation_delay(leo2dt_distance))

    feedback_overhead = (agent.sat.trans_latency(len(self.ues) * constant.FLOAT_SIZE, self.dt_server)
                         + util.propagation_delay(leo2dt_distance))

    overhead = (max(realworld_header, digitalworld_header)
                + agent.computation_latency + feedback_overhead)

    if self.last_episode:
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
