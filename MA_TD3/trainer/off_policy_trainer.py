"""off_policy_trainer.py"""
from __future__ import annotations
from logging import Logger
from collections import OrderedDict
from typing import Dict, List, Tuple
import copy
import time
import random
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import numpy.typing as npt

from gym_env.leosat.leosat_env import LEOSatEnv
from gym_env.digitalworld.digitalworld_env import DigitalWorldEnv
from gym_env.realworld.realworld_env import RealWorldEnv
from ..agent import Agent
from ..misc import misc


class OffPolicyTrainer(object):
  """The trainer class"""

  def __init__(self, args: Namespace, log: Dict[str, Logger], tb_writer: SummaryWriter, env: DigitalWorldEnv | RealWorldEnv, leo_agent_dict: Dict[str, Agent], online=False):
    self.args = args
    self.log = log
    self.tb_writer = tb_writer
    self.env = env
    self.__online = online
    self.__twin_trainer = None
    self.total_timesteps = 0  # steps of collecting experience
    self.total_train_iter = 0  # steps of training iteration
    self.total_eps = 0  # steps of episodes
    self.leo_agent_dict = leo_agent_dict
    self.agent_num = len(self.leo_agent_dict)

    self.__cur_states = {}
    for agent_name, agent in self.leo_agent_dict.items():
      self.__cur_states[agent_name] = [0] * agent.state_dim

    self.parameter_db = {}
    for agent_name in leo_agent_dict.keys():
      self.parameter_db[agent_name] = {}
    self.weight_db = {}

    self.sat_sim_time = 0
    self.nn_train_time = 0
    self.param_sharing_time = 0
    self.total_training_time = 0
    self.nn_action_time = 0
    self.tb_time = 0
    self.init_time = 0

  @property
  def total_eps(self):
    return self._total_eps

  @total_eps.setter
  def total_eps(self, eps):
    self._total_eps = eps

  @property
  def twin_trainer(self) -> OffPolicyTrainer:
    return self.__twin_trainer

  @twin_trainer.setter
  def twin_trainer(self, trainer: OffPolicyTrainer):
    if not self.__twin_trainer:
      self.__twin_trainer = trainer
    else:
      raise ValueError(f'{self.twin_trainer.env.unwrapped.name} is gonna replace by {trainer.env.unwrapped.name}')

  @property
  def cur_states(self):
    if not self.online:
      raise ValueError(f'{self.env.unwrapped.name} is not online.')
    return self.__cur_states

  @cur_states.setter
  def cur_states(self, state):
    self.__cur_states = state

  @property
  def online(self):
    return self.__online

  @online.setter
  def online(self, online):
    if type(online) is not bool:
      raise TypeError('online can only be bool.')
    if online is True:
      # self.copy_NN_from_twin()
      self.total_eps = self.twin_trainer.total_eps
    self.__online = online

  def combined_state(self, sat_name) -> npt.NDArray[np.float32]:
    if not self.online:
      raise ValueError(f'{self.env.unwrapped.name} is offline.')
    if not self.twin_trainer.online:
      twin_state_len = self.twin_trainer.leo_agent_dict[sat_name].self_state_dim
      state_len = self.leo_agent_dict[sat_name].self_state_dim
      if twin_state_len > state_len:
        return np.concatenate((self.cur_states[sat_name], self.cur_states[sat_name], np.zeros((twin_state_len - state_len,))))
      else:
        return np.concatenate((self.cur_states[sat_name], self.cur_states[sat_name][:twin_state_len]))
    else:
      return np.concatenate((self.cur_states[sat_name], self.twin_trainer.cur_states[sat_name]))
    


  def twin_parameter_sharing(self):
    if not self.twin_trainer.online:
      print(f'{self.twin_trainer.env.unwrapped.name} is not online. No NN is copied to {self.twin_trainer.env.unwrapped.name}')
    else:
      for sat_name in self.leo_agent_dict:
        pretrained_actor, pretrained_critic = self.twin_trainer.leo_agent_dict[sat_name].model_state_dict
        self.leo_agent_dict[sat_name].load_actor_state_dict(pretrained_actor)
        self.leo_agent_dict[sat_name].load_critic_state_dict(pretrained_critic)

  def federated_upload(self):
    """Federated uploading for the models of the given agents."""
    ps_start_time = time.time()

    for agent_name, agent in self.leo_agent_dict.items():
      if not self.parameter_db[agent_name]:
        self.parameter_db[agent_name]['actor'] = copy.deepcopy(
          agent.actor_state_dict)
        self.parameter_db[agent_name]['critic'] = copy.deepcopy(
          agent.critic_state_dict)
      else:
        if self.args.partial_upload_type == 'random':
          a_rand_idx = random.sample(range(0, int(agent.actor_layer_num)),
                                     self.args.uploaded_layer_num_per_turn)
          c_rand_idx = random.sample(range(0, int(agent.critic_layer_num)),
                                     self.args.uploaded_layer_num_per_turn)

        elif self.args.partial_upload_type == 'by-turns':
          # 0-index
          a_rand_idx, agent.cur_actorlayer_idx = misc.circ_range(agent.cur_actorlayer_idx,
                                                                 self.args.uploaded_layer_num_per_turn,
                                                                 agent.actor_layer_num)
          c_rand_idx, agent.cur_criticlayer_idx = misc.circ_range(agent.cur_criticlayer_idx,
                                                                  self.args.uploaded_layer_num_per_turn,
                                                                  agent.actor_layer_num)

        else:
          raise ValueError(
            f'No \"{self.args.partial_upload_type}\" partial upload type.')

        cur_actor_state_dict, cur_critic_state_dict = agent.model_state_dict

        # Upload a random layer of the actor network to the central server
        a_upload_dict = OrderedDict()
        for a_idx in a_rand_idx:
          key = f'network.fc_{a_idx}.weight'
          a_upload_dict[key] = copy.deepcopy(cur_actor_state_dict[key])
          key = f'network.fc_{a_idx}.bias'
          a_upload_dict[key] = copy.deepcopy(cur_actor_state_dict[key])

        self.parameter_db[agent_name]['actor'].update(a_upload_dict)

        # Upload a random layer of the critic network to the central server
        c_upload_dict = OrderedDict()
        for c_idx in c_rand_idx:
          for qi in range(agent.q_network_num):
            key = f'q{qi + 1}_network.fc_{c_idx}.weight'
            c_upload_dict[key] = copy.deepcopy(cur_critic_state_dict[key])
            key = f'q{qi + 1}_network.fc_{c_idx}.bias'
            c_upload_dict[key] = copy.deepcopy(cur_critic_state_dict[key])

        self.parameter_db[agent_name]['critic'].update(c_upload_dict)

      self.weight_db[agent_name] = agent.sharing_weight

    self.param_sharing_time += time.time() - ps_start_time

  def federated_download(self):
    """Federated dowloading for the models of the given agents."""
    ps_start_time = time.time()

    # Calculate the sum parameters
    actor_sum_sd, critic_sum_sd = OrderedDict(), OrderedDict()  # state_dict
    total_weight = 0.0
    for agent_name, database_sd in self.parameter_db.items():
      if len(database_sd) != 0:
        param_weight = self.weight_db[agent_name]
        total_weight += param_weight
        if not actor_sum_sd:
          for key, item in database_sd['actor'].items():
            actor_sum_sd[key] = torch.mul(item, param_weight)
          for key, item in database_sd['critic'].items():
            critic_sum_sd[key] = torch.mul(item, param_weight)
        else:
          for key, item in database_sd['actor'].items():
            actor_sum_sd[key] += torch.mul(item, param_weight)
          for key, item in database_sd['critic'].items():
            critic_sum_sd[key] += torch.mul(item, param_weight)

    # Update the parameters to the requested agents
    for agent_name, agent in self.leo_agent_dict.items():
      tau = agent.federated_update_rate
      actor_sd, critic_sd = agent.model_state_dict

      # Update actor parameters to local agent
      for key, item in actor_sd.items():
        if key not in actor_sum_sd:
          raise ValueError(
            f'Key error in actor database storing (key = {key}).')
        actor_sd[key] = torch.add(torch.mul(item, 1 - tau),
                                  torch.mul(actor_sum_sd[key], tau / total_weight))

      # Update critic parameters to local agent
      for key, item in critic_sd.items():
        if key not in critic_sum_sd:
          raise ValueError(
            f'Key error in critic database storing (key = {key}).')
        critic_sd[key] = torch.add(torch.mul(item, 1 - tau),
                                   torch.mul(critic_sum_sd[key], tau / total_weight))

      agent.load_actor_state_dict(actor_sd)
      agent.load_critic_state_dict(critic_sd)

    self.param_sharing_time += time.time() - ps_start_time

  def train(self):
    """
    1. Train the neural network.
    2. Parameter sharing.
    """
    if not self.online:
      print(f'{self.env.unwrapped.name} is offline')
      return
    nn_start_time = time.time()

    if self.total_timesteps % self.args.training_period == 0:
      self.total_train_iter += 1
      for _, agent in self.leo_agent_dict.items():
        # Update policy (iteration of training is args.iter_num)
        agent.update_policy(self.total_train_iter)

    self.nn_train_time += time.time() - nn_start_time

    if self.total_timesteps % self.args.federated_upload_period == 0:
      self.federated_upload()

    if self.total_timesteps % self.args.federated_download_period == 0:
      self.federated_download()

  def print_time(self):
    self.total_training_time = (self.sat_sim_time + self.nn_train_time +
                                self.param_sharing_time + self.nn_action_time + self.init_time + self.tb_time)
    if self.total_training_time == 0:
      return
    print('------------------------------')
    print(f'{self.env.unwrapped.name}:')
    print(
      f'Satellite simulation time ratio: {self.sat_sim_time / self.total_training_time * 100:.2f} %')
    print(
      f'NN training time ratio: {self.nn_train_time / self.total_training_time * 100:.2f} %')
    print(
      f'Parameter sharing time ratio: {self.param_sharing_time / self.total_training_time * 100:.2f} %')
    print(
      f'Initialize Env time ratio: {self.init_time / self.total_training_time * 100:.2f} %')
    print(
      f'Action computation time ratio: {self.nn_action_time / self.total_training_time * 100:.2f} %')
    print(
      f'Tensorboard saving time ratio: {self.tb_time / self.total_training_time * 100:.2f} %')
    print(f'total running time: {self.total_training_time / 3600: .2f} hr')
    print('------------------------------')

  def save_eval_result(self, step_count: int) -> Dict[str, float]:
    if not self.online:
      return

    start_time = time.time()
    self.env.unwrapped.save_episode_result()
    for agent_name in self.ep_reward:
      self.ep_reward[agent_name] /= step_count
    for agent_name, agent in self.leo_agent_dict.items():
      self.log[self.args.log_name].info(
          f'Agent {agent.name}: Evaluation Reward {self.ep_reward[agent_name]:.6f} at episode {self.total_eps}')
      self.tb_writer.add_scalars(
        'Eval_reward', {f'{self.env.unwrapped.name} {agent_name} reward': self.ep_reward[agent_name]}, self.total_eps)
    self.tb_writer.add_scalars(
      'Eval_reward', {f'{self.env.unwrapped.name} total reward': sum(self.ep_reward.values())}, self.total_eps)
    self.tb_time += time.time() - start_time

  def save_training_result(self, step_count: int):
    if not self.online:
      return
    start_time = time.time()
    for agent_name in self.ep_reward:
      self.ep_reward[agent_name] /= step_count
    self.total_eps += 1
    for agent_name, agent in self.leo_agent_dict.items():
      self.log[self.args.log_name].info(
          f'Agent {agent.name}: Training episode reward {self.ep_reward[agent_name]:.6f} at episode {self.total_eps}')
      self.tb_writer.add_scalars(
        f'{agent.name}/training_reward', {'training_reward': self.ep_reward[agent_name]}, self.total_eps)
    self.tb_time += time.time() - start_time

  def take_action(self, action_dict, running_mode='training') -> Tuple[Dict[str, npt.NDArray[np.float32]], float, bool]:
    if not self.online:
      return None, None, False
    # Take action in env
    sim_start_time = time.time()

    new_env_observation, env_reward, done, _, _ = self.env.step(action_dict)
    self.total_timesteps += 1

    if running_mode == "training":
      self.train()
    else:
      self.env.render()

    # For next timesteps
    prev_state_dict = {}
    for agent_name in self.leo_agent_dict:
      prev_state_dict[agent_name] = self.combined_state(agent_name)
    self.cur_states = new_env_observation
    for agent_name in env_reward:
      self.ep_reward[agent_name] += env_reward[agent_name]

    self.sat_sim_time += time.time() - sim_start_time

    return prev_state_dict, sum(env_reward.values()), done

  def no_action_step(self, running_mode='training'):
    if not self.online:
      return None, None, None, False
    # Take action in env
    sim_start_time = time.time()

    new_env_observation, env_reward, done, _, _ = self.env.no_action_step()

    if running_mode == "testing":
      self.env.render()

    # For next timesteps
    self.cur_states = new_env_observation
    self.total_timesteps += 1
    for agent_name in env_reward:
      self.ep_reward[agent_name] += env_reward[agent_name]

    self.sat_sim_time += time.time() - sim_start_time

    return sum(env_reward.values()), done

  def save_to_replaybuffer(self, prev_state_dict, action_dict, total_reward, done):
    if prev_state_dict is None or action_dict is None:
      return
    for agent_name, leo_agent in self.leo_agent_dict.items():
      leo_agent.add_memory(
          obs=prev_state_dict[agent_name],
          new_obs=self.combined_state(agent_name),
          action=action_dict[agent_name],
          reward=total_reward,
          done=done)

  def stochastic_actions(self) -> Dict[str, npt.NDArray[np.float32]]:
    start_time = time.time()
    if not self.online:
      return None
    action_dict = {}
    for agent_name, leo_agent in self.leo_agent_dict.items():
      agent_action = leo_agent.select_stochastic_action(
        np.asarray(self.combined_state(agent_name)), self.total_timesteps)
      action_dict[agent_name] = agent_action

    self.nn_action_time += time.time() - start_time
    return action_dict

  def deterministic_actions(self) -> Dict[str, npt.NDArray[np.float32]]:
    start_time = time.time()
    if not self.online:
      return None

    action_dict = {}
    for agent_name, leo_agent in self.leo_agent_dict.items():
      agent_action = leo_agent.select_deterministic_action(
        np.asarray(self.combined_state(agent_name)))
      action_dict[agent_name] = agent_action

    self.nn_action_time += time.time() - start_time
    return action_dict

  def reset_env(self, eval=False):
    if not self.online:
      return
    if self.total_eps > self.args.max_ep_num - 1 and eval:
      self.env.unwrapped.last_episode = True
    else:
      self.env.unwrapped.last_episode = False
    start_time = time.time()

    self.env.unwrapped.set_online(self_online=self.online, twin_online=self.twin_trainer.online)
    self.ep_reward = {}
    for agent_name in self.leo_agent_dict:
      self.ep_reward[agent_name] = 0.0
    self.cur_states, _ = self.env.reset()

    self.init_time += time.time() - start_time
