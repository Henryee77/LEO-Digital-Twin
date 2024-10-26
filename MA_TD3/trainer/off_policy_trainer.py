"""off_policy_trainer.py"""
from __future__ import annotations
from logging import Logger
from collections import OrderedDict
from typing import Dict, List, Tuple
import copy
import time
import random
from argparse import Namespace
from tensorboardX import SummaryWriter
import torch
import numpy as np
import numpy.typing as npt

from gym_env.leosat.leosat_env import LEOSatEnv
from ..agent import Agent
from ..misc import misc


class OffPolicyTrainer(object):
  """The trainer class"""

  def __init__(self, args: Namespace, log: Dict[str, Logger], tb_writer: SummaryWriter, env: LEOSatEnv, leo_agent_dict: Dict[str, Agent]):
    self.args = args
    self.log = log
    self.tb_writer = tb_writer
    self.env = env
    self.total_timesteps = 0  # steps of collecting experience
    self.total_train_iter = 0  # steps of training iteration
    self.total_eps = 0  # steps of episodes
    self.leo_agent_dict = leo_agent_dict
    self.agent_num = len(self.leo_agent_dict)

    self.cur_states = {}
    for agent_name, agent in self.leo_agent_dict.items():
      self.cur_states[agent_name] = [0] * agent.state_dim

    self.parameter_db = {}
    for agent_name in leo_agent_dict.keys():
      self.parameter_db[agent_name] = {}
    self.weight_db = {}

    self.sat_sim_time = 0
    self.nn_train_time = 0
    self.eval_time = 0
    self.param_sharing_time = 0
    self.total_training_time = 0

  @property
  def total_eps(self):
    return self._total_eps

  @property
  def cur_states(self):
    return self.__cur_states

  @cur_states.setter
  def cur_states(self, state):
    self.__cur_states = state

  @total_eps.setter
  def total_eps(self, eps):
    self._total_eps = eps

  def combined_state(self, sat_name) -> npt.NDArray[np.float32]:
    return np.concatenate((self.cur_states[sat_name], self.twin_trainer.cur_states[sat_name]))

  def set_twin_trainer(self, twin_trainer: OffPolicyTrainer):
    self.twin_trainer = twin_trainer

  def drop_twin_trainer(self):
    self.twin_trainer = None

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
            key = f'q{qi+1}_network.fc_{c_idx}.weight'
            c_upload_dict[key] = copy.deepcopy(cur_critic_state_dict[key])
            key = f'q{qi+1}_network.fc_{c_idx}.bias'
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
    nn_start_time = time.time()

    for _ in range(self.args.iter_num):
      self.total_train_iter += 1
      for _, agent in self.leo_agent_dict.items():
        # Update policy (iteration of training is args.iter_num)
        agent.update_policy(self.total_train_iter)

    self.nn_train_time += time.time() - nn_start_time

    if self.total_eps % self.args.federated_upload_freq == 0:
      self.federated_upload()

    if self.total_eps % self.args.federated_download_freq == 0:
      self.federated_download()

  def print_time(self):
    self.total_training_time = self.sat_sim_time + self.nn_train_time + self.param_sharing_time + self.eval_time
    print(
      f'Satellite simulation time ratio: {self.sat_sim_time / self.total_training_time * 100 :.2f} %')
    print(
      f'NN training time ratio: {self.nn_train_time / self.total_training_time * 100 :.2f} %')
    print(
      f'Parameter sharing time ratio: {self.param_sharing_time / self.total_training_time * 100 :.2f} %')
    print(
      f'Evaluation time ratio: {self.eval_time / self.total_training_time * 100 :.2f} %')
    print(f'total running time: {self.total_training_time / 3600: .2f} hr')

  def save_eval_result(self, step_count: int) -> Dict[str, float]:
    for agent_name in self.ep_reward:
      self.ep_reward[agent_name] /= step_count
    for agent_name, agent in self.leo_agent_dict.items():
      self.log[self.args.log_name].info(
          f'Agent {agent.name}: Evaluation Reward {self.ep_reward[agent_name]:.6f} at episode {self.total_eps}')
      self.tb_writer.add_scalars(
        'Eval_reward', {f'{self.env.name} {agent_name} reward': self.ep_reward[agent_name]}, self.total_eps)
    self.tb_writer.add_scalars(
      'Eval_reward', {f'{self.env.name} total reward': sum(self.ep_reward.values())}, self.total_eps)

  def save_training_result(self, step_count: int):
    for agent_name in self.ep_reward:
      self.ep_reward[agent_name] /= step_count
    self.total_eps += 1
    for agent_name, agent in self.leo_agent_dict.items():
      self.log[self.args.log_name].info(
          f'Agent {agent.name}: Training episode reward {self.ep_reward[agent_name]:.6f} at episode {self.total_eps}')
      self.tb_writer.add_scalars(
        f'{agent.name}/training_reward', {'training_reward': self.ep_reward[agent_name]}, self.total_eps)

  def take_action(self, action_dict, running_mode='training') -> Tuple[Dict[str, npt.NDArray[np.float32]], Dict[str, npt.NDArray[np.float32]], float, bool]:
    # Take action in env
    sim_start_time = time.time()

    new_env_observation, env_reward, done, _, _ = self.env.step(action_dict)

    if running_mode == "testing":
      self.env.render()

    # For next timestep
    prev_state_dict = {}
    for agent_name in self.leo_agent_dict:
      prev_state_dict[agent_name] = self.combined_state(agent_name)
    self.cur_states = new_env_observation
    self.total_timesteps += 1
    for agent_name in env_reward:
      self.ep_reward[agent_name] += env_reward[agent_name]

    self.sat_sim_time += time.time() - sim_start_time

    return prev_state_dict, action_dict, sum(env_reward.values()), done

  def save_to_replaybuffer(self, prev_state_dict, action_dict, total_reward, done):
    for agent_name, leo_agent in self.leo_agent_dict.items():
      leo_agent.add_memory(
          obs=prev_state_dict[agent_name],
          new_obs=self.combined_state(agent_name),
          action=action_dict[agent_name],
          reward=total_reward,
          done=done)

  def stochastic_actions(self) -> Dict[str, npt.NDArray[np.float32]]:
    action_dict = {}
    for agent_name, leo_agent in self.leo_agent_dict.items():
      agent_action = leo_agent.select_stochastic_action(
        np.asarray(self.combined_state(agent_name)), self.total_timesteps)
      action_dict[agent_name] = agent_action
    return action_dict

  def deterministic_actions(self) -> Dict[str, npt.NDArray[np.float32]]:
    eval_start_time = time.time()

    action_dict = {}
    for agent_name, leo_agent in self.leo_agent_dict.items():
      agent_action = leo_agent.select_deterministic_action(
        np.asarray(self.combined_state(agent_name)))
      action_dict[agent_name] = agent_action

    self.eval_time = time.time() - eval_start_time
    return action_dict

  def reset_env(self):
    self.ep_reward = {}
    for agent_name in self.leo_agent_dict:
      self.ep_reward[agent_name] = 0.0
    self.cur_states, _ = self.env.reset()
