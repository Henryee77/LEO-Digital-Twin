"""utils.py for NN training"""
import csv
import copy
import math
import logging
from logging import Logger
from datetime import datetime
from collections import OrderedDict
from typing import List, Tuple, Dict, OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import gymnasium as gym
from gymnasium import spaces
import yaml
import git
import numpy as np
import matplotlib.pyplot as plt
from gym_env.leosat import LEOSatEnv
import gym_env  # this line is neccessary, don't delete it.

from MA_TD3.agent.agent import Agent
from low_earth_orbit.util import constant

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_config(args, path='.'):
  now = datetime.now()
  time_string = now.strftime("%Y_%m_%d_%H_%M_%S")
  with open(f'{path}/config/{args.prefix}_{time_string}', mode='w', encoding='utf-8') as f:
    yaml.safe_dump(args.__dict__, f)


def load_config(args, path='.'):
  """Loads and replaces default parameters with experiment
  specific parameters

  Args:
      args (argparse): Python argparse that contains arguments
      path (str): Root directory to load config from. Default: "."
  """
  with open(f'{path}/config/{args.prefix}', mode='r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

  for key, value in config.items():
    args.__dict__[key] = value


def set_logger(logger_name, log_file, level=logging.INFO):
  """Sets python logging

  Args:
      logger_name (str): Specifies logging name
      log_file (str): Specifies path to save logging
      level (int): Logging when above specified level. Default: logging.INFO
  """
  log = logging.getLogger(logger_name)
  formatter = logging.Formatter('%(asctime)s : %(message)s')
  print(f'log_file: {log_file}')
  fileHandler = logging.FileHandler(log_file, mode='a')
  fileHandler.setFormatter(formatter)
  streamHandler = logging.StreamHandler()
  streamHandler.setFormatter(formatter)

  log.setLevel(level)
  log.addHandler(fileHandler)
  log.addHandler(streamHandler)


def set_log(args, path=".") -> Dict[str, Logger]:
  """Loads and replaces default parameters with experiment specific parameters.

  Args:
      args (argparse): Python argparse that contains arguments
      path (str): Root directory to get Git repository. Default: "."

  Examples:
      log[args.log_name].info("Hello {}".format("world"))

  Returns:
      log (dict): Dictionary that contains python logging
  """
  log = {}
  set_logger(
      logger_name=args.log_name,
      log_file=r'{0}{1}'.format('./log/', args.log_name))
  log[args.log_name] = logging.getLogger(args.log_name)

  for arg, value in sorted(vars(args).items()):
    log[args.log_name].info("%s: %r", arg, value)

  repo = git.Repo(path)
  log[args.log_name].info('Branch: {}'.format(repo.active_branch))
  log[args.log_name].info('Commit: {}'.format(repo.head.commit))

  return log


def make_env(env_name, args, ax: plt.Axes, tb_writer: SummaryWriter, agent_dict: Dict[str, Agent], real_agents: Dict[str, Agent], digital_agents: Dict[str, Agent], agent_names: List[str]) -> LEOSatEnv:
  env = gym.make(env_name,
                 ax=ax,
                 args=args,
                 tb_writer=tb_writer,
                 agent_dict=agent_dict,
                 real_agents=real_agents,
                 digital_agents=digital_agents,
                 agent_names=agent_names)

  return env


def circ_range(start: int, num: int, modulo: int) -> Tuple[List[int], int]:
  """Return the circular space of modulo

  Args:
      start (int): start value.
      num (int): number of the list.
      modulo (int): modulo.

  Returns:
      List[int]: [start, start + 1, ..., start + num] % modulo
      int: The next index.
  """
  result = []
  index = start
  for _ in range(num):
    result.append(index)
    index = int((index + 1) % modulo)
  return result, index


def load_rt_file(filename: str) -> Dict[str, Dict[int, Dict[str, float]]]:
  """Load the ray tracing simulation result file.
  ### Nested dictionary hierarchy:
    {t: {sat_name: {beam_index: [ray tracing data 1, ..., ray tracing data N]}}}
  ### Key of ray tracing data: 
    - ue
    - received power (W)
    - phase (radians)
    - path loss (dB)
    - path gain (dB)
    - h_r
    - h_i
  ### Example:
    ray_tracing_data = misc.load_rt_file()
    target_ue_path_loss = [data['path loss (dB)'] for data in ray_tracing_data[t][sat_name][beam_index] if data['ue'] == target_ue]
  """

  rt_result = {}
  with open(f'MA_TD3/misc/rt_result/{filename}.csv', mode='r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
      sat_name = row.pop('sat_name')
      t = int(row.pop('t'))
      b_i = int(row.pop('beam_index'))
      if t not in rt_result:
        rt_result[t] = {}
      if sat_name not in rt_result[t]:
        rt_result[t][sat_name] = {}
      if b_i not in rt_result[t][sat_name]:
        rt_result[t][sat_name][b_i] = []

      for key in row:
        if key == 'ue':
          row[key] = int(row[key]) - 1
        else:
          row[key] = float(row[key])
      if math.isinf(row['path loss (dB)']):
        row['path loss (dB)'] = constant.MAX_DB
        row['path gain (dB)'] = constant.MIN_DB
      rt_result[t][sat_name][b_i].append(row)

  return rt_result


def construct_dnn_dict(input_dim: int, output_dim: int, hidden_nodes: List[int], activ_func) -> OrderedDict[str, nn.Module]:
  layer_list = []
  assert hidden_nodes[-1] > output_dim
  layer_nodes = [input_dim] + hidden_nodes + [output_dim]
  for i in range(len(layer_nodes) - 1):
    # print(f'fc_{i}', layer_nodes[i], layer_nodes[i + 1])
    layer_list.append((f'fc_{i}', nn.Linear(layer_nodes[i], layer_nodes[i + 1])))
    if i != len(layer_nodes) - 2:
      layer_list.append((f'activ_{i}', copy.deepcopy(activ_func)))

  return OrderedDict(layer_list)


def generate_action_space(cell_num: int):
  beam_action_low = np.array([-1] * cell_num)
  beam_action_high = np.array([1] * cell_num)
  beam_slice = slice(0, cell_num)

  total_power_low = np.array([-1])
  total_power_high = np.array([1])
  power_slice = slice(0, cell_num + 1)

  beamwidth_action_low = np.array([-1] * cell_num)
  beamwidth_action_high = np.array([1] * cell_num)
  beamwidth_slice = slice(cell_num + 1, cell_num * 2 + 1)

  action_low = np.concatenate(
      (beam_action_low, total_power_low, beamwidth_action_low))

  action_high = np.concatenate(
      (beam_action_high, total_power_high, beamwidth_action_high))

  action_space = spaces.Box(low=np.float32(action_low),
                            high=np.float32(action_high),
                            dtype=np.float32)

  return action_space, beam_slice, power_slice, beamwidth_slice


def generate_state_space(agent_type: str, cell_num: int, pos_dim: int = 2):
  pos_low = np.array([-1] * pos_dim)
  pos_high = np.array([1] * pos_dim)
  pos_slice = slice(0, pos_dim)

  r_beam_info_low = np.array([-1] * cell_num)
  r_beam_info_high = np.array([1] * cell_num)
  r_obs_low = np.concatenate((pos_low, r_beam_info_low))
  r_obs_high = np.concatenate((pos_high, r_beam_info_high))

  d_beam_info_low = np.array([-1] * (cell_num * 3))
  d_beam_info_high = np.array([1] * (cell_num * 3))
  d_obs_low = np.concatenate((pos_low, d_beam_info_low))
  d_obs_high = np.concatenate((pos_high, d_beam_info_high))

  if agent_type == 'real_LEO':
    obs_low = np.concatenate((r_obs_low, d_obs_low))
    obs_high = np.concatenate((r_obs_high, d_obs_high))
    beam_info_slice = slice(pos_dim, pos_dim + cell_num)
  elif agent_type == 'digital_LEO':
    obs_low = np.concatenate((d_obs_low, r_obs_low))
    obs_high = np.concatenate((d_obs_high, r_obs_high))
    beam_info_slice = slice(pos_dim, pos_dim + (cell_num * 3))
  else:
    raise ValueError('No such agent type')

  observation_space = spaces.Box(low=np.float32(obs_low),
                                 high=np.float32(obs_high),
                                 dtype=np.float32)

  return observation_space, pos_slice, beam_info_slice
