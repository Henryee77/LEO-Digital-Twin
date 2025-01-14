"""off_policy_trainer.py"""
from __future__ import annotations
from logging import Logger
from collections import OrderedDict
from typing import Dict, List, Tuple, Any
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
from low_earth_orbit.util import util
from ..misc import misc


class OffPolicyTrainer(object):
  """The trainer class"""

  parameter_db: Dict[str, Dict[str, OrderedDict]]

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
    self.agent_name_list = list(leo_agent_dict.keys())
    self.agent_num = len(self.leo_agent_dict)

    self.__cur_states = {}
    for agent_name, agent in self.leo_agent_dict.items():
      self.__cur_states[agent_name] = [0] * agent.state_dim

    self.parameter_db = {}
    self.twin_nn = {}
    for agent_name in leo_agent_dict.keys():
      self.parameter_db[agent_name] = {}
      self.twin_nn[agent_name] = {}
    self.weight_db = {}

    self.sat_sim_time = 0
    self.nn_train_time = 0
    self.federated_sharing_time = 0
    self.twin_sharing_time = 0
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
    if self.__twin_trainer:
      raise ValueError(f'{self.twin_trainer.env.unwrapped.name} is gonna replaced by {trainer.env.unwrapped.name}')
    elif self is trainer:
      raise ValueError(f'{self.env.unwrapped.name} is same as the assigned twin trainer {trainer.env.unwrapped.name}')
    else:
      self.__twin_trainer = trainer

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

  def aggregated_state_dict(self) -> Dict[str, npt.NDArray[np.float32]]:
    if self.args.scope_of_states == 'local':
      return {agent_name: self.dt_combined_state(agent_name) for agent_name in self.agent_name_list}
    elif self.args.scope_of_states == 'global':
      all_states = [self.dt_combined_state(agent_name) for agent_name in self.agent_name_list]
      return {agent_name: np.hstack(all_states) for agent_name in self.agent_name_list}

  def dt_combined_state(self, sat_name) -> npt.NDArray[np.float32]:
    if not self.online:
      raise ValueError(f'{self.env.unwrapped.name} is offline.')
    if not self.twin_trainer.online:
      # TODO This section has the risk of causing bug, be careful!
      twin_state_len = self.twin_trainer.leo_agent_dict[sat_name].self_state_dim
      state_len = self.leo_agent_dict[sat_name].self_state_dim
      if twin_state_len > state_len:  # I'm real and the DTs are offline.
        return np.concatenate((self.cur_states[sat_name], self.cur_states[sat_name], np.zeros((twin_state_len - state_len,))))
      else:  # I'm DT and the real LEOs are offline.
        return np.concatenate((self.cur_states[sat_name], self.cur_states[sat_name][:twin_state_len]))
    else:
      # TODO Using the name attribute to identify the trainer may cause some bugs, be careful!
      if self.env.unwrapped.name == 'Real World':
        real_state = self.cur_states[sat_name]

        digital_agent = self.twin_trainer.leo_agent_dict[sat_name]
        shared_digital_state = self.twin_trainer.cur_states[sat_name][digital_agent.shared_slice]
        padding_len = digital_agent.self_state_dim - len(shared_digital_state)
        digital_state = np.concatenate((shared_digital_state, np.zeros(padding_len,)))
      elif self.env.unwrapped.name == 'Digital World':
        digital_state = self.cur_states[sat_name]
        real_state = self.twin_trainer.cur_states[sat_name]
      else:
        raise ValueError(f'No such {self.env.unwrapped.name} trainer and env.')

      return np.concatenate((real_state, digital_state))

  def receive_sensing_env_param(self):
    """Set the parameters of the virtual environment"""
    if self.env.unwrapped.name == 'Real World':
      raise ValueError('Cannot change the real world environment parameters by copying DT\'s value')

    real_agents = self.twin_trainer.leo_agent_dict
    error = self.args.dt_param_error

    # Sharing LEO params
    for sat_name in self.leo_agent_dict:
      self.leo_agent_dict[sat_name].sat.nakagami_m = (
        (1 + error * util.random_sign()) * real_agents[sat_name].sat.nakagami_m)
      self.leo_agent_dict[sat_name].sat.los_power_ratio = (
        (1 + error * util.random_sign()) * real_agents[sat_name].sat.los_power_ratio)
      self.leo_agent_dict[sat_name].sat.rx_power_ratio = (
        (1 + error * util.random_sign()) * real_agents[sat_name].sat.rx_power_ratio)

    # Sharing ground params
    real_ue_dict = self.twin_trainer.env.unwrapped.ue_dict
    digital_dict = self.env.unwrapped.ue_dict
    assert real_ue_dict is not digital_dict
    for ue_name in real_ue_dict:
      digital_dict[ue_name].water_vap_density = (
        (1 + error * util.random_sign()) * real_ue_dict[ue_name].water_vap_density)
      digital_dict[ue_name].temperature = (
        (1 + error * util.random_sign()) * real_ue_dict[ue_name].temperature)
      digital_dict[ue_name].atmos_pressure = (
        (1 + error * util.random_sign()) * real_ue_dict[ue_name].atmos_pressure)

  def twin_parameter_query(self):
    if not self.online or not self.twin_trainer.online:
      return
    ps_start_time = time.time()

    if not self.twin_trainer.online:
      print(f'{self.twin_trainer.env.unwrapped.name} is not online.')
      return
    else:
      for agent_name in self.leo_agent_dict:
        agent = self.leo_agent_dict[agent_name]
        twin_agent = self.twin_trainer.leo_agent_dict[agent_name]
        twin_actor, twin_critic = twin_agent.model_state_dict
        a_idx_list, c_idx_list = misc.agent_sharing_layer(self.args, agent, self.args.twin_sharing_layer_num_per_turn)
        agent.set_twin_sharing_param_num(a_idx_list, c_idx_list)

        agent.twin_sharing_actor = misc.copy_layers_from_actor(twin_actor, a_idx_list)
        agent.twin_sharing_critic = misc.copy_layers_from_critic(twin_critic, c_idx_list, agent.q_network_num)

    self.twin_sharing_time += time.time() - ps_start_time

  def twin_parameter_update(self):
    if not self.online or not self.twin_trainer.online:
      return
    ps_start_time = time.time()

    for agent in self.leo_agent_dict.values():
      agent.update_nn_from_twin_sharing()

    self.twin_sharing_time += time.time() - ps_start_time

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
        a_idx_list, c_idx_list = misc.agent_sharing_layer(self.args, agent, self.args.federated_layer_num_per_turn)

        cur_actor_state_dict, cur_critic_state_dict = agent.model_state_dict

        # Upload a random layer of the actor network to the central server
        a_upload_dict = misc.copy_layers_from_actor(cur_actor_state_dict, a_idx_list)
        self.parameter_db[agent_name]['actor'].update(a_upload_dict)

        # Upload a random layer of the critic network to the central server
        c_upload_dict = misc.copy_layers_from_critic(cur_critic_state_dict, c_idx_list, agent.q_network_num)

        self.parameter_db[agent_name]['critic'].update(c_upload_dict)

      self.weight_db[agent_name] = agent.sharing_weight

    self.federated_sharing_time += time.time() - ps_start_time

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

    self.federated_sharing_time += time.time() - ps_start_time

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

    if self.env.unwrapped.name == self.args.digital_env_name:
      if self.total_timesteps % self.args.federated_upload_period == 0:
        self.federated_upload()
      if self.total_timesteps % self.args.federated_download_period == 0:
        self.federated_download()

  def print_time(self):
    self.total_training_time = (self.sat_sim_time + self.nn_train_time +
                                self.federated_sharing_time + self.twin_sharing_time +
                                self.nn_action_time + self.init_time + self.tb_time)
    if self.total_training_time == 0:
      return
    self.log[self.args.log_name].info('------------------------------')
    self.log[self.args.log_name].info(f'{self.env.unwrapped.name}:')
    self.log[self.args.log_name].info(
      f'Satellite simulation time ratio: {self.sat_sim_time / self.total_training_time * 100:.2f} %')
    self.log[self.args.log_name].info(
      f'NN training time ratio: {self.nn_train_time / self.total_training_time * 100:.2f} %')
    self.log[self.args.log_name].info(
      f'Federated sharing time ratio: {self.federated_sharing_time / self.total_training_time * 100:.2f} %')
    self.log[self.args.log_name].info(
      f'Twin sharing time ratio: {self.twin_sharing_time / self.total_training_time * 100:.2f} %')
    self.log[self.args.log_name].info(
      f'Initialize Env time ratio: {self.init_time / self.total_training_time * 100:.2f} %')
    self.log[self.args.log_name].info(
      f'Action computation time ratio: {self.nn_action_time / self.total_training_time * 100:.2f} %')
    self.log[self.args.log_name].info(
      f'Tensorboard saving time ratio: {self.tb_time / self.total_training_time * 100:.2f} %')
    self.log[self.args.log_name].info(f'total running time: {self.total_training_time / 3600: .2f} hr')
    self.log[self.args.log_name].info('------------------------------')

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
        'Eval_reward/agent reward', {f'{self.env.unwrapped.name} {agent_name} reward': self.ep_reward[agent_name]}, self.total_eps)
    self.tb_writer.add_scalars(
      'Eval_reward/total reward', {f'{self.env.unwrapped.name} total reward': sum(self.ep_reward.values())}, self.total_eps)
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

  def take_action(self, action_dict, running_mode='training') -> Tuple[Dict[str, npt.NDArray[np.float32]], Dict[str, float], bool, Dict[Any, Any]]:
    if not self.online:
      return None, None, False, None
    # Take action in env
    sim_start_time = time.time()

    new_env_observation, env_reward, done, _, info = self.env.step(action_dict)

    self.sat_sim_time += time.time() - sim_start_time

    if running_mode == "training":
      self.train()
    else:
      self.env.render()

    # For next timesteps
    prev_state_dict = {}
    aggr_state_dict = self.aggregated_state_dict()
    for agent_name in self.leo_agent_dict:
      prev_state_dict[agent_name] = aggr_state_dict[agent_name]
    self.cur_states = new_env_observation
    self.total_timesteps += 1
    for agent_name in env_reward:
      self.ep_reward[agent_name] += env_reward[agent_name]

    return prev_state_dict, env_reward, done, info

  def no_action_step(self, running_mode='training'):
    if not self.online:
      return None, False, None
    # Take action in env
    sim_start_time = time.time()

    new_env_observation, env_reward, done, _, info = self.env.unwrapped.no_action_step()

    self.sat_sim_time += time.time() - sim_start_time

    if running_mode == "training":
      self.train()
    else:
      self.env.render()

    # For next timesteps
    self.cur_states = new_env_observation
    self.total_timesteps += 1
    for agent_name in env_reward:
      self.ep_reward[agent_name] += env_reward[agent_name]

    return sum(env_reward.values()), done, info

  def save_to_replaybuffer(self,
                           agent_type: str,
                           prev_state_dict: Dict[str, npt.NDArray[np.float32]],
                           action_dict: Dict[str, npt.NDArray[np.float32]],
                           reward_dict: Dict[str, float],
                           done: bool):
    if prev_state_dict is None or action_dict is None:
      return
    aggr_state_dict = self.aggregated_state_dict()
    if agent_type == 'DT':
      total_reward = sum(reward_dict.values())
      for agent_name in reward_dict:
        reward_dict[agent_name] = total_reward
    elif agent_type == 'LEO':
      pass
    else:
      raise ValueError(f'No such {agent_type} type of agents.')
    for agent_name, leo_agent in self.leo_agent_dict.items():
      leo_agent.add_memory(
          obs=prev_state_dict[agent_name],
          new_obs=aggr_state_dict[agent_name],
          action=action_dict[agent_name],
          reward=reward_dict[agent_name],
          done=done)

  def stochastic_actions(self) -> Dict[str, npt.NDArray[np.float32]]:
    start_time = time.time()
    if not self.online:
      return None
    action_dict = {}
    aggr_state_dict = self.aggregated_state_dict()
    for agent_name, leo_agent in self.leo_agent_dict.items():
      agent_action = leo_agent.select_stochastic_action(
        np.asarray(aggr_state_dict[agent_name]), self.total_timesteps)
      action_dict[agent_name] = agent_action

    self.nn_action_time += time.time() - start_time
    return action_dict

  def deterministic_actions(self) -> Dict[str, npt.NDArray[np.float32]]:
    start_time = time.time()
    if not self.online:
      return None

    action_dict = {}
    aggr_state_dict = self.aggregated_state_dict()
    for agent_name, leo_agent in self.leo_agent_dict.items():
      agent_action = leo_agent.select_deterministic_action(
        np.asarray(aggr_state_dict[agent_name]))
      action_dict[agent_name] = agent_action

    self.nn_action_time += time.time() - start_time
    return action_dict

  def reset_env(self, eval=False) -> Dict[Any, Any]:
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
    self.cur_states, info = self.env.reset()

    self.init_time += time.time() - start_time

    return info
