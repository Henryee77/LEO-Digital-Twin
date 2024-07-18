"""agent.py"""
import math
from typing import Literal
import numpy as np
from misc.replay_buffer import ReplayBuffer
from policy.model import TD3, DDPG
from low_earth_orbit.util import constant


class Agent(object):
  def __init__(self, env, policy_name: Literal['TD3', 'DDPG'], tb_writer, log, args, name, device):

    self.env = env
    self.log = log
    self.tb_writer = tb_writer
    self.args = args
    self.name = name
    self.device = device

    self.set_dim()
    self.set_policy(policy_name)
    self.memory = ReplayBuffer(max_size=args.replay_buffer_size)
    self.epsilon = 1  # For exploration
    self.max_action = self.env.action_space.high[0]
    self.min_action = self.env.action_space.low[0]
    self.sharing_weight = 1
    self.beta = args.historical_smoothing_coef
    self.beta_pow_n = self.beta
    self.historical_avg_reward = 0

    self._federated_update_rate = args.federated_update_rate
    self.cur_actorlayer_idx = 1
    self.cur_criticlayer_idx = 1

  def set_dim(self):
    self.state_dim = self.env.observation_space.shape[0]
    self.action_dim = self.env.action_space.shape[0]

    self.log[self.args.log_name].info('[{}] State dim: {}'.format(
        self.name, self.state_dim))
    self.log[self.args.log_name].info('[{}] Action dim: {}'.format(
        self.name, self.action_dim))

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

  def clear_memory(self):
    self.memory.clear()

  def set_policy(self, policy_name):
    if policy_name == 'TD3':
      self.policy = TD3(
          actor_input_dim=self.state_dim,
          actor_output_dim=self.action_dim,
          critic_input_dim=self.state_dim + self.action_dim,
          name=self.name,
          args=self.args,
          action_low=self.env.action_space.low,
          action_high=self.env.action_space.high,
          device=self.device)
    elif policy_name == 'DDPG':
      self.policy = DDPG(state_dim=self.state_dim,
                         action_dim=self.action_dim,
                         device=self.device,
                         args=self.args)
    else:
      raise ValueError(f'No {policy_name} policy')

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
      noise = np.random.normal((self.max_action + self.min_action) / 2,
                               (self.max_action - self.min_action) * self.args.expl_noise,
                               size=len(action))

      assert np.size(action) == np.size(noise)
      action = np.clip(action + noise,
                       self.env.action_space.low,
                       self.env.action_space.high)
      # print(f'ep-greedy: {action}')
    else:
      action = self.env.action_space.sample()
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
    # self.beta_pow_n *= math.pow(self.beta, self.args.iter_num * self.args.federated_freq)
    self.historical_avg_reward = self.beta * self.historical_avg_reward + (1 - self.beta) * r
    # / (1 - self.beta_pow_n))

    self.sharing_weight = min(self.args.max_sharing_weight, r / self.historical_avg_reward)

    self.tb_writer.add_scalars(f'{self.name}/historical_avg_reward',
                               {self.name: self.historical_avg_reward}, total_train_iter)
    self.tb_writer.add_scalars(f'{self.name}/sharing_weight',
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
          f'{self.name}/{key}', {self.name: value}, total_train_iter)
