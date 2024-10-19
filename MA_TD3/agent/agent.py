"""agent.py"""
from __future__ import annotations
from typing import Literal, Dict
from logging import Logger
from gymnasium import spaces
from torch import device
from tensorboardX import SummaryWriter
import numpy as np
import numpy.typing as npt
from ..misc.replay_buffer import ReplayBuffer
from ..policy.model import TD3, DDPG
from low_earth_orbit.util.position import Position, Geodetic
from low_earth_orbit.util import constant
from low_earth_orbit.satellite.satellite import Satellite
from low_earth_orbit.cell.cell_topology import CellTopology
from low_earth_orbit.antenna.antenna import Antenna
from low_earth_orbit.util import util


class Agent(object):
  def __init__(self,
               policy_name: Literal['TD3', 'DDPG'],
               tb_writer: SummaryWriter,
               log: Dict[str, Logger],
               args,
               name: str,
               agent_type: str,
               device: device,
               comp_freq: float = constant.DEFAULT_CPU_CYCLE):

    self.log = log
    self.tb_writer = tb_writer
    self.args = args
    self.name = name
    self.agent_type = agent_type
    self.device = device
    self.comp_freq = comp_freq
    self.sat = Satellite(shell_index=0,
                         plane_index=0,
                         sat_index=0,
                         angle_speed=0,
                         position=Position(geodetic=Geodetic(0, 0, constant.R_EARTH)),
                         cell_topo=CellTopology(center_point=Position(geodetic=Geodetic(0, 0, constant.R_EARTH))),
                         antenna=Antenna(),
                         channel=None)

    if agent_type == 'real_LEO':
      pass
    elif agent_type == 'digital_LEO':
      self._init_dim()
      self.set_policy(policy_name)
    else:
      raise ValueError('No such agent type')

    self.memory = ReplayBuffer(max_size=args.replay_buffer_size)
    self.epsilon = 1  # For exploration
    self.sharing_weight = 1
    self.beta = args.historical_smoothing_coef
    self.beta_pow_n = self.beta
    self.historical_avg_reward = 0

    self._federated_update_rate = args.federated_update_rate
    self.cur_actorlayer_idx = 1
    self.cur_criticlayer_idx = 1

  def _init_dim(self):
    cell_num = self.sat.cell_topo.cell_number
    beam_action_low = np.array([-1] * cell_num)
    beam_action_high = np.array([1] * cell_num)
    self.beam_slice = slice(0, cell_num)

    self.total_power_low = np.array([-1])
    self.total_power_high = np.array([1])
    self.min_power = 40
    self.max_power = self.sat.max_power
    self.power_slice = slice(0, cell_num + 1)

    self.beamwidth_action_low = np.array([-1] * cell_num)
    self.beamwidth_action_high = np.array([1] * cell_num)
    self.beamwidth_slice = slice(cell_num + 1, cell_num * 2 + 1)
    self.min_beamwidth = 2 * constant.PI_IN_RAD
    self.max_beamwidth = 4 * constant.PI_IN_RAD

    action_low = np.concatenate(
      (beam_action_low, self.total_power_low, self.beamwidth_action_low))

    action_high = np.concatenate(
      (beam_action_high, self.total_power_high, self.beamwidth_action_high))

    self.action_space = spaces.Box(low=np.float32(action_low),
                                   high=np.float32(action_high),
                                   dtype=np.float32)

    self.pos_low = np.array([-1] * 2)
    self.pos_high = np.array([1] * 2)
    sinr_diff_low = np.array([-1] * cell_num)
    sinr_diff_high = np.array([1] * cell_num)

    obs_low = np.concatenate((self.pos_low, sinr_diff_low))
    obs_high = np.concatenate((self.pos_high, sinr_diff_high))

    self.observation_space = spaces.Box(low=np.float32(obs_low),  #
                                        high=np.float32(obs_high),
                                        dtype=np.float32)

    self.max_actions = self.action_space.high
    self.min_actions = self.action_space.low

    self.state_dim = self.observation_space.shape[0]
    self.action_dim = self.action_space.shape[0]
    self.actor_n_hidden = self.args.ra_actor_n_hidden
    self.critic_n_hidden = self.args.ra_critic_n_hidden

    self.log[self.args.log_name].info('[{}] State dim: {}'.format(
        self.name, self.state_dim))
    self.log[self.args.log_name].info('[{}] Action dim: {}'.format(
        self.name, self.action_dim))

  def set_policy(self, policy_name):
    denom = [1, 2, 4, 8, 16, 32]
    self.actor_hidden_nodes = [round(self.actor_n_hidden / x) for x in denom]
    self.critic_hidden_nodes = [round(self.critic_n_hidden / x) for x in denom]
    if policy_name == 'TD3':
      self.policy = TD3(
          actor_input_dim=self.state_dim,
          actor_output_dim=self.action_dim,
          critic_input_dim=self.state_dim + self.action_dim,
          actor_hidden_nodes=self.actor_hidden_nodes,
          critic_hidden_nodes=self.critic_hidden_nodes,
          name=self.name,
          args=self.args,
          action_low=self.action_space.low,
          action_high=self.action_space.high,
          device=self.device)
    elif policy_name == 'DDPG':
      self.policy = DDPG(state_dim=self.state_dim,
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
  def action_dim(self):
    return self._action_dim

  @action_dim.setter
  def action_dim(self, dim):
    if not isinstance(dim, int):
      raise TypeError('Action dimension must be int')
    self._action_dim = dim

  @property
  def state_dim(self):
    return self._state_dim

  @state_dim.setter
  def state_dim(self, dim):
    if not isinstance(dim, int):
      raise TypeError('State dimension must be int')
    self._state_dim = dim

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
    return self._federated_update_rate

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
      w = constant.MIN_POSITIVE_FLOAT
    self._sharing_weight = w

  @property
  def clear_memory(self):
    self.memory.clear()

  @property
  def computation_latency(self) -> float:
    return constant.F_0 * self.nn_param_num / self.comp_freq

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
    self.policy.load_critic_state_dict(state_dict=state_dict)

  def save_weight(self, filename, directory):
    self.log[self.args.log_name].info("[{}] Saved weight".format(self.name))
    self.policy.save(filename, directory)

  def load_weight(self, filename, directory="./pytorch_models"):
    """Load state dict from file.

    Args:
        filename (_type_): filename
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
        f'debug/{self.agent_type} {self.name}_epsilon', self.epsilon, total_timesteps)

    return action

  def add_memory(self, obs, new_obs, action, reward, done):
    self.memory.add((obs, new_obs, action, reward, done))

  def update_share_weight(self, r: float, total_train_iter: int):
    # self.beta_pow_n *= math.pow(self.beta, self.args.iter_num * self.args.federated_freq)
    self.historical_avg_reward = self.beta * self.historical_avg_reward + (1 - self.beta) * r
    # / (1 - self.beta_pow_n))

    self.sharing_weight = min(self.args.max_sharing_weight, r / self.historical_avg_reward)

    self.tb_writer.add_scalars(f'{self.agent_type} {self.name}/historical_avg_reward',
                               {self.name: self.historical_avg_reward}, total_train_iter)
    self.tb_writer.add_scalars(f'{self.agent_type} {self.name}/sharing_weight',
                               {self.name: self.sharing_weight}, total_train_iter)

  def update_policy(self, total_train_iter):
    if len(self.memory) > self.args.batch_size * self.args.iter_num * 2:
      debug = self.policy.train(
          replay_buffer=self.memory,
          total_train_iter=total_train_iter)

      # federated
      if 'actor_loss' in debug:
        self.update_share_weight(-1 * debug['actor_loss'].item(), total_train_iter)

      for key, value in debug.items():
        self.tb_writer.add_scalars(
          f'{self.agent_type} {self.name}/{key}', {self.name: value}, total_train_iter)
