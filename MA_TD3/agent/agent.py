"""agent.py"""
from __future__ import annotations
from typing import Literal, Dict
from logging import Logger
import queue
from gymnasium import spaces
from torch import device
from torch.utils.tensorboard import SummaryWriter
from argparse import Namespace
import torch
import numpy as np
import numpy.typing as npt
from ..misc.replay_buffer import ReplayBuffer
from ..misc import misc
from ..policy.model import TD3, DDPG
from low_earth_orbit.util.position import Position, Geodetic
from low_earth_orbit.util import constant
from low_earth_orbit.satellite.satellite import Satellite
from low_earth_orbit.cell.cell_topology import CellTopology
from low_earth_orbit.channel import Channel
from low_earth_orbit.util import util


class Agent(object):
  def __init__(self,
               policy_name: Literal['TD3', 'DDPG'],
               tb_writer: SummaryWriter,
               log: Dict[str, Logger],
               args: Namespace,
               sat_name: str,
               agent_type: str,
               device: device,
               comp_freq: float,
               total_agent_num: int):

    self.log = log
    self.tb_writer = tb_writer
    self.args = args
    self.agent_type = agent_type
    self.device = device
    self.comp_freq = comp_freq
    self.total_agent_num = total_agent_num
    sat_indices = sat_name.split('_')
    self.sat = Satellite(shell_index=sat_indices[0],
                         plane_index=sat_indices[1],
                         sat_index=sat_indices[2],
                         angle_speed=0,
                         position=Position(geodetic=Geodetic(0, 0, constant.R_EARTH)),
                         cell_topo=CellTopology(center_point=Position(geodetic=Geodetic(0, 0, constant.R_EARTH)),
                                                cell_layer=args.cell_layer_num),
                         channel=Channel(rain_prob=self.args.rainfall_prob,
                                         has_weather=self.args.has_weather_module)
                         )

    self._init_dim()
    self.set_policy(policy_name)

    self.memory = ReplayBuffer(max_size=args.replay_buffer_size)
    self.epsilon = 1  # For exploration
    self.sharing_weight = 1
    self.hist_reward_q = queue.Queue()
    self.hist_total_reward = 0

    self.cur_actorlayer_idx = 1
    self.cur_criticlayer_idx = 1
    self.twin_sharing_actor = None
    self.twin_sharing_critic = None
    self.twin_sharing_param_num = 0

  def _init_dim(self):
    if self.args.scope_of_states == 'local':
      received_agent_num = 1
    elif self.args.scope_of_states == 'global':
      received_agent_num = self.total_agent_num
    else:
      raise ValueError(f'No {self.args.scope_of_states} type of --scope-of-states')
    if self.args.scope_of_actions == 'distributed':
      decided_agent_num = 1
    elif self.args.scope_of_actions == 'centralized':
      decided_agent_num = self.total_agent_num
    else:
      raise ValueError(f'No {self.args.scope_of_states} type of --scope-of-states')

    (self.action_space,
     self.beam_slice,
     self.power_slice,
     self.beamwidth_slice) = misc.generate_action_space(self.sat.cell_topo.cell_number,
                                                        decided_agent_num=decided_agent_num)

    (self.observation_space,
     self.pos_slice,
     self.beam_info_slice,
     self.shared_slice) = misc.generate_state_space(agent_type=self.agent_type,
                                                    cell_num=self.sat.cell_topo.cell_number,
                                                    shared_type=self.args.shared_state_type,
                                                    received_agent_num=received_agent_num)

    self.min_actions = self.action_space.low
    self.max_actions = self.action_space.high

    self.__self_state_dim = (len(self.observation_space.low[self.pos_slice]) +
                             len(self.observation_space.low[self.beam_info_slice]))
    self.__state_dim = self.observation_space.shape[0]
    self.__action_dim = self.action_space.shape[0]

    self.log[self.args.log_name].info('[{}] State dim: {}'.format(
        self.name, self.state_dim))
    self.log[self.args.log_name].info('[{}] Action dim: {}'.format(
        self.name, self.action_dim))

  def set_policy(self, policy_name):
    hidden_layer_denom = [1, 2, 4, 8, 16]
    self.actor_hidden_nodes = [round(self.args.actor_n_hidden / x) for x in hidden_layer_denom]
    self.critic_hidden_nodes = [round(self.args.critic_n_hidden / x) for x in hidden_layer_denom]
    if policy_name == 'TD3':
      self.__policy = TD3(
          actor_input_dim=self.state_dim,
          actor_output_dim=self.action_dim,
          critic_input_dim=self.state_dim + self.action_dim,
          actor_hidden_nodes=self.actor_hidden_nodes,
          critic_hidden_nodes=self.critic_hidden_nodes,
          name=self.name,
          args=self.args,
          action_low=self.min_actions,
          action_high=self.max_actions,
          device=self.device)
    elif policy_name == 'DDPG':
      self.__policy = DDPG(state_dim=self.state_dim,
                           action_dim=self.action_dim,
                           device=self.device,
                           args=self.args)
    else:
      raise ValueError(f'No {policy_name} policy')

  @property
  def sat(self) -> Satellite:
    return self._sat

  @sat.setter
  def sat(self, sat: Satellite):
    self._sat = sat

  @property
  def sat_name(self) -> str:
    return self.sat.name

  @property
  def name(self) -> str:
    return self.agent_type + '_' + self.sat_name

  @property
  def policy(self):
    return self.__policy

  @policy.setter
  def policy(self, pol):
    self.__policy = pol

  @property
  def action_dim(self):
    return self.__action_dim

  @property
  def state_dim(self):
    return self.__state_dim

  @property
  def self_state_dim(self):
    return self.__self_state_dim

  @property
  def pos_low(self):
    return self.observation_space.low[self.pos_slice]

  @property
  def pos_high(self):
    return self.observation_space.high[self.pos_slice]

  @property
  def total_power_low(self):
    return self.action_space.low[self.power_slice][-1]

  @property
  def total_power_high(self):
    return self.action_space.high[self.power_slice][-1]

  @property
  def beamwidth_action_low(self):
    return self.action_space.low[self.beamwidth_slice]

  @property
  def beamwidth_action_high(self):
    return self.action_space.high[self.beamwidth_slice]

  @property
  def actor_state_dict(self):
    return self.policy.actor.state_dict()

  @property
  def critic_state_dict(self):
    return self.policy.critic.state_dict()

  @property
  def model_state_dict(self):
    return self.actor_state_dict, self.critic_state_dict

  @property
  def actor_layer_num(self):
    return self.policy.actor_layer_num

  @property
  def critic_layer_num(self):
    return self.policy.critic_layer_num

  @property
  def q_network_num(self):
    return self.policy.q_network_num

  @property
  def nn_param_num(self):
    return self.policy.nn_param_num

  @property
  def federated_update_rate(self):
    return self.args.federated_update_rate

  @property
  def twin_sharing_update_rate(self):
    return self.args.twin_sharing_update_rate

  @property
  def cur_actorlayer_idx(self):
    """The current uploading layer index of the actor network"""
    return self._cur_actorlayer_idx

  @cur_actorlayer_idx.setter
  def cur_actorlayer_idx(self, i):
    self._cur_actorlayer_idx = i

  @property
  def cur_criticlayer_idx(self):
    """The current uploading layer index of the critic network"""
    return self._cur_criticlayer_idx

  @cur_criticlayer_idx.setter
  def cur_criticlayer_idx(self, i):
    self._cur_criticlayer_idx = i

  @property
  def sharing_weight(self):
    return self._sharing_weight

  @sharing_weight.setter
  def sharing_weight(self, w):
    if w < 0:
      self.log[self.args.log_name].info(f'Invalid weight of agent {self.name} (w = {w})')
      # print(self.historical_avg_reward)
      w = constant.MIN_POSITIVE_FLOAT
    self._sharing_weight = w

  @property
  def clear_memory(self):
    self.memory.clear()

  @property
  def computation_latency(self) -> float:
    return constant.F_0 * self.nn_param_num / self.comp_freq

  @property
  def twin_sharing_param_num(self):
    return self.__twin_sharing_param_num

  @twin_sharing_param_num.setter
  def twin_sharing_param_num(self, num):
    self.__twin_sharing_param_num = num

  def set_twin_sharing_param_num(self, a_idx_list, c_idx_list):
    self.twin_sharing_param_num = (sum([self.policy.actor_layer_param_num[a_idx] for a_idx in a_idx_list]) +
                                   sum([self.policy.critic_layer_param_num[c_idx] for c_idx in c_idx_list]))

  def get_scaled_pos(self, plot_range: float) -> npt.NDArray[np.float32]:
    return np.array([util.rescale_value(self.sat.position.geodetic.longitude,
                                        constant.ORIGIN_LONG - plot_range,
                                        constant.ORIGIN_LONG + plot_range,
                                        self.pos_low[0],
                                        self.pos_high[0]),
                     util.rescale_value(self.sat.position.geodetic.latitude,
                                        constant.ORIGIN_LATI - plot_range,
                                        constant.ORIGIN_LATI + plot_range,
                                        self.pos_low[1],
                                        self.pos_high[1])])

  def load_actor_state_dict(self, state_dict):
    self.policy.load_actor_state_dict(state_dict)

  def load_critic_state_dict(self, state_dict):
    self.policy.load_critic_state_dict(state_dict)

  def update_nn_from_twin_sharing(self):
    tau = self.twin_sharing_update_rate
    for key in self.twin_sharing_actor:
      self.actor_state_dict[key] = torch.add(torch.mul(self.actor_state_dict[key], 1 - tau),
                                             torch.mul(self.twin_sharing_actor[key], tau))

    for key in self.twin_sharing_critic:
      self.critic_state_dict[key] = torch.add(torch.mul(self.critic_state_dict[key], 1 - tau),
                                              torch.mul(self.twin_sharing_critic[key], tau))

  def save_weight(self, filename, directory):
    self.log[self.args.log_name].info("[{}] Saved weight".format(self.name))
    self.policy.save(filename, directory)

  def load_weight(self, filename: str, directory="./pytorch_models"):
    """Load state dict from file.

    Args:
        filename (str): filename
        directory (str, optional): directory of the file. Defaults to "./pytorch_models".
    """
    self.log[self.args.log_name].info("[{}] Loaded weight".format(self.name))
    self.policy.load(filename, directory)

  def select_deterministic_action(self, obs):
    action = self.policy.select_action(obs)
    assert not np.isnan(action).any()

    return action

  def select_stochastic_action(self, obs, total_timesteps):
    if np.random.rand() > self.epsilon:
      action = self.policy.select_action(obs)
      noise = np.array([np.random.normal((max_a + min_a) / 2, (max_a - min_a) * self.args.expl_noise)
                       for max_a, min_a in zip(self.max_actions, self.min_actions)])

      assert np.size(action) == np.size(noise)
      action = np.clip(action + noise,
                       self.action_space.low,
                       self.action_space.high)
      # print(f'ep-greedy: {action}')
    else:
      action = self.action_space.sample()
      # print(f'ep-greedy: {action}')

    if self.epsilon > self.args.min_epsilon and total_timesteps > self.args.full_explore_steps:
      self.epsilon *= self.args.epsilon_decay_rate  # Reduce epsilon over time

    if np.isnan(action).any():
      print(f'obs: {obs}')
      print(f'action: {action}')

    assert not np.isnan(action).any()

    self.tb_writer.add_scalar(
        f'debug/{self.name}_epsilon', self.epsilon, total_timesteps)

    return action

  def add_memory(self, obs, new_obs, action, reward, done):
    self.memory.add((obs, new_obs, action, reward, done))

  def update_share_weight(self, r: float, total_train_iter: int):
    if self.hist_reward_q.qsize() >= self.args.historical_reward_window:
      self.hist_total_reward -= self.hist_reward_q.get(block=False)
    self.hist_total_reward += r
    self.hist_reward_q.put(r)

    historical_avg_reward = self.hist_total_reward / self.hist_reward_q.qsize()

    indic = (r - historical_avg_reward) / historical_avg_reward
    beta = self.args.sharing_weight_growth_rate
    nu = self.args.sharing_weight_asymptote_occurrence

    self.sharing_weight = (1 + np.exp(-beta * indic)) ** (-nu)

    self.tb_writer.add_scalars(f'{self.name}/historical_avg_reward',
                               {self.name: historical_avg_reward}, total_train_iter)
    self.tb_writer.add_scalars(f'{self.name}/sharing_weight',
                               {self.name: self.sharing_weight}, total_train_iter)

  def update_policy(self, total_train_iter):
    if len(self.memory) > self.args.batch_size * 2:
      debug = self.policy.train(
          replay_buffer=self.memory,
          total_train_iter=total_train_iter)

      # federated
      if 'actor_loss' in debug:
        self.update_share_weight(-1 * debug['actor_loss'].item(), total_train_iter)

      for key, value in debug.items():
        self.tb_writer.add_scalars(
          f'{self.name}/{key}', {self.name: value}, total_train_iter)
