"""utils.py for NN training"""
import logging
from datetime import datetime
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import gymnasium as gym
import yaml
import git
import numpy as np
import matplotlib.pyplot as plt
import gym_env  # this line is neccessary


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


def set_log(args, path="."):
  """Loads and replaces default parameters with experiment
  specific parameters

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


def make_env(args, ax: plt.Axes, agent_names) -> gym.Env:
  env = gym.make(args.env_name, ax=ax, args=args, agent_names=agent_names)

  return env


def onehot_from_logits(logits):
  """Given batch of logits, return one-hot sample
  Ref: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/misc.py
  """
  if isinstance(logits, np.ndarray):
    logits = torch.from_numpy(logits)
    if len(logits.shape) == 1:
      logits = logits.unsqueeze(0)  # Add a dimension

  argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
  return argmax_acs


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
  """Sample from Gumbel(0, 1)
  Ref: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
  Ref: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/misc.py
  """
  U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
  return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution
  Ref: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
  Ref: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/misc.py
  """
  if device == torch.device('cuda'):
    tens_type = torch.cuda.FloatTensor
  elif device == torch.device('cpu'):
    tens_type = torch.FloatTensor
  else:
    raise ValueError('Invalid dtype')

  y = logits + sample_gumbel(logits.shape, tens_type=tens_type)
  return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Ref: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
  Ref: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/misc.py
  """
  if isinstance(logits, np.ndarray):
    logits = torch.from_numpy(logits)
    if len(logits.shape) == 1:
      logits = logits.unsqueeze(0)  # Add a dimension
  assert len(logits.shape) == 2, "Shape should be: (# of batch, # of action)"

  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    y_hard = onehot_from_logits(y)
    y = (y_hard - y).detach() + y

  return y


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
