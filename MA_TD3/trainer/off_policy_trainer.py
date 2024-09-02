"""off_policy_trainer.py"""
import copy
import time
import random
from collections import OrderedDict
from typing import Dict, List
import torch
import numpy as np

from ..agent import Agent
from ..misc import misc


class OffPolicyTrainer(object):
  """The trainer class"""

  def __init__(self, args, leo_agent_dict: Dict[str, Agent], channel_agent):
    self.args = args
    self.total_timesteps = 0  # steps of collecting experience
    self.total_train_iter = 0  # steps of training iteration
    self.total_eps = 0  # steps of episodes
    self.leo_agent_dict = leo_agent_dict
    self.agent_num = len(self.leo_agent_dict)

    self.parameter_db = {}
    for agent_name in leo_agent_dict.keys():
      self.parameter_db[agent_name] = {}
    self.weight_db = {}

    self.sat_sim_time = 0
    self.nn_train_time = 0
    self.eval_time = 0
    self.param_sharing_time = 0

  @property
  def total_eps(self):
    return self.total_eps

  def federated_upload(self, agent_names: List[str]):
    """Federated uploading for the models of the given agents.

    Args:
        agent_names (List[str]): The names of agents perform the FU.
    """
    for agent_name in agent_names:
      agent = self.leo_agent_dict[agent_name]
      if not self.parameter_db[agent_name]:
        self.parameter_db[agent_name]['actor'] = copy.deepcopy(
          agent.actor_state_dict)
        self.parameter_db[agent_name]['critic'] = copy.deepcopy(
          agent.critic_state_dict)
      else:
        if self.args.partial_upload_type == 'random':
          a_rand_idx = random.sample(range(1, int(agent.actor_layer_num) + 1),
                                     self.args.uploaded_layer_num_per_turn)
          c_rand_idx = random.sample(range(1, int(agent.critic_layer_num) + 1),
                                     self.args.uploaded_layer_num_per_turn)

        elif self.args.partial_upload_type == 'by-turns':
          # 0-index
          a_rand_idx, agent.cur_actorlayer_idx = misc.circ_range(agent.cur_actorlayer_idx,
                                                                 self.args.uploaded_layer_num_per_turn,
                                                                 agent.actor_layer_num)
          c_rand_idx, agent.cur_criticlayer_idx = misc.circ_range(agent.cur_criticlayer_idx,
                                                                  self.args.uploaded_layer_num_per_turn,
                                                                  agent.actor_layer_num)
          # 1-index
          # a_rand_idx = [round(idx + 1) for idx in a_rand_idx]
          # c_rand_idx = [round(idx + 1) for idx in c_rand_idx]

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

  def federated_download(self, agent_names: List[str]):
    """Federated dowloading for the models of the given agents.

    Args:
        agent_names (List[str]): The names of agents perform the FD.
    """
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
    for agent_name in agent_names:
      agent = self.leo_agent_dict[agent_name]
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

  def train(self, env, log, tb_writer):
    sim_start_time = time.time()
    self.collect_one_traj(env=env, log=log, tb_writer=tb_writer)
    self.sat_sim_time += time.time() - sim_start_time

    for _ in range(self.args.iter_num):
      self.total_train_iter += 1
      nn_start_time = time.time()
      for _, agent in self.leo_agent_dict.items():
        # Update policy (iteration of training is args.iter_num)
        agent.update_policy(self.total_train_iter)
      self.nn_train_time += time.time() - nn_start_time

      if self.total_eps % self.args.federated_upload_freq == 0:
        ps_start_time = time.time()
        self.federated_upload(agent_names=list(self.leo_agent_dict.keys()))
        self.param_sharing_time += time.time() - ps_start_time

      if self.total_eps % self.args.federated_download_freq == 0:
        ps_start_time = time.time()
        self.federated_download(agent_names=list(self.leo_agent_dict.keys()))
        self.param_sharing_time += time.time() - ps_start_time

      # Measure performance  # or (self.total_eps <= 20 and self.total_eps % 2 == 0)
      eval_start_time = time.time()
      if self.total_eps % self.args.eval_freq == 0:
        self.eval_progress(env=env, log=log, tb_writer=tb_writer)
      self.eval_time = time.time() - eval_start_time

  def print_time(self):
    total_training_time = self.sat_sim_time + self.nn_train_time + self.param_sharing_time + self.eval_time
    print(
      f'Satellite simulation time ratio: {self.sat_sim_time / total_training_time * 100 :.2f} %')
    print(
      f'NN training time ratio: {self.nn_train_time / total_training_time * 100 :.2f} %')
    print(
      f'Parameter sharing time ratio: {self.param_sharing_time / total_training_time * 100 :.2f} %')
    print(
      f'Evaluation time ratio: {self.eval_time / total_training_time * 100 :.2f} %')
    print(f'total running time: {total_training_time / 3600: .2f} hr')

  def eval_progress(self, env, log, tb_writer, running_mode='training'):
    eval_reward = {}
    for agent_name in self.leo_agent_dict:
      eval_reward[agent_name] = 0.0
    step_count = 0
    env_observation, _ = env.reset()

    while True:
      # print(f'{ep_timesteps}, {eval_reward}')
      # Select action
      # print(f'obs: {env_observation}, {type(env_observation)}')
      action_n = {}
      for agent_name, agent in self.leo_agent_dict.items():
        agent_action = agent.select_deterministic_action(
            np.array(env_observation[agent_name]))
        action_n[agent_name] = agent_action
      # print(f'act: {agent_action}')
      # Take action in env
      new_env_observation, env_reward, done, truncated, _ = env.step(
          copy.deepcopy(action_n))
      # print(f'new obs: {new_env_observation}')
      # For next timestep
      env_observation = new_env_observation
      for agent_name in env_reward:
        eval_reward[agent_name] += env_reward[agent_name]
      step_count += 1

      if running_mode == "testing":
        env.render()

      # TODO
      # Need to be modified to multi-agent ver.
      if running_mode == 'testing':
        beam_power_actions = agent_action[env.power_slice]
        for i, action in enumerate(beam_power_actions):
          if i < env.cell_num:
            tb_writer.add_scalars('actions/power ratio',
                                  {f'beam {i}': action}, step_count)
          else:
            tb_writer.add_scalars('actions/total power',
                                  {'total power': action}, step_count)

        beamwidth_action = agent_action[env.beamwidth_slice]
        for i, action in enumerate(beamwidth_action):
          tb_writer.add_scalars('actions/beamwidth',
                                {f'beam {i}': action}, step_count)

      if done or truncated:
        break
    for agent_name in eval_reward:
      eval_reward[agent_name] /= step_count
    for agent_name in self.leo_agent_dict:
      log[self.args.log_name].info(
          f'Agent {agent_name}: Evaluation Reward {eval_reward[agent_name]:.6f} at episode {self.total_eps}')
      tb_writer.add_scalars(
        'Eval_reward', {f'{agent_name} reward': eval_reward[agent_name]}, self.total_eps)
    tb_writer.add_scalars(
      'Eval_reward', {'total reward': sum(eval_reward.values())}, self.total_eps)

  def collect_one_traj(self, env, log, tb_writer):
    ep_reward = {}
    for agent_name in self.leo_agent_dict:
      ep_reward[agent_name] = 0.0
    step_count = 0
    env_observation, _ = env.reset()

    while True:
      # Select action
      # print(f'obs: {env_observation}')
      action_n = {}
      for agent_name, leo_agent in self.leo_agent_dict.items():
        agent_action = leo_agent.select_stochastic_action(
          np.array(env_observation[agent_name]), self.total_timesteps)
        action_n[agent_name] = agent_action

      # Take action in env
      new_env_observation, env_reward, done, truncated, _ = env.step(action_n)

      # Add experience to memory
      total_reward = sum(env_reward.values())
      for agent_name, leo_agent in self.leo_agent_dict.items():
        leo_agent.add_memory(
            obs=env_observation[agent_name],
            new_obs=new_env_observation[agent_name],
            action=action_n[agent_name],
            reward=total_reward,
            done=done)

      # For next timestep
      env_observation = new_env_observation
      step_count += 1
      self.total_timesteps += 1
      for agent_name in env_reward:
        ep_reward[agent_name] += env_reward[agent_name]

      if done or truncated:
        break

    for agent_name in ep_reward:
      ep_reward[agent_name] /= step_count
    self.total_eps += 1
    for agent_name in self.leo_agent_dict:
      log[self.args.log_name].info(
          f'Agent {agent_name}: Train episode reward {ep_reward[agent_name]:.6f} at episode {self.total_eps}')
      tb_writer.add_scalars(
        f'{agent_name}/reward', {'train_reward': ep_reward[agent_name]}, self.total_eps)

    return ep_reward

  def test(self, env, log, tb_writer):
    self.eval_progress(env=env,
                       log=log,
                       tb_writer=tb_writer,
                       running_mode='testing')
