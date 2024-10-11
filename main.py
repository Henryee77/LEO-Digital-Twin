"""main.py"""
import os
import argparse
import time
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from MA_TD3.misc import misc
from MA_TD3.agent import Agent
from MA_TD3.trainer import OffPolicyTrainer


def main(args):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(device)
  # Create directories
  if not os.path.exists('./log'):
    os.makedirs('./log')
  if not os.path.exists('./pytorch_models'):
    os.makedirs('./pytorch_models')
  if not os.path.exists('./config'):
    os.makedirs('./config')

  # Set logs
  tb_writer = SummaryWriter(f'./log/tb_{args.log_name}')
  log = misc.set_log(args)
  saving_directory = 'pytorch_models'
  loading_directory = 'pytorch_models'
  filename = args.model

  # Set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # Initialize visualization
  fig = plt.figure(figsize=(16, 9), dpi=80)
  fig.set_tight_layout(True)
  ax = plt.axes()
  ax.set_aspect('equal', adjustable='box')
  plt.ion()

  # Create env
  real_env = misc.make_env(args.real_env_name, args=args, ax=ax, agent_names=agent_name_list)
  digital_env = misc.make_env(args.digital_env_name, args=args, ax=ax, agent_names=agent_name_list)

  # Initialize agents
  agent_name_list = ['3_0_24', '2_0_1', '1_0_9']
  realworld_agent_dict = {}
  digitalworld_agent_dict = {}
  for agent_name in agent_name_list:
    realworld_agent_dict[agent_name] = Agent(env=real_env,
                                             policy_name=args.model,
                                             tb_writer=tb_writer,
                                             log=log,
                                             name=agent_name,
                                             agent_type='LEO',
                                             args=args,
                                             device=device)
    digitalworld_agent_dict[agent_name] = Agent(env=digital_env,
                                                policy_name=args.model,
                                                tb_writer=tb_writer,
                                                log=log,
                                                name=agent_name,
                                                agent_type='LEO',
                                                args=args,
                                                device=device)
  real_env.leo_agents = realworld_agent_dict
  digital_env.leo_agents = digitalworld_agent_dict
  # Start training
  trainer_dict = {'TD3': 'Off-Policy',
                  'DDPG': 'Off-Policy',
                  'A3C': 'Asynchronous_On-Policy'}

  # Off-Policy
  if trainer_dict[args.model] == 'Off-Policy':
    realworld_trainer = OffPolicyTrainer(args=args,
                                         log=log,
                                         tb_writer=tb_writer,
                                         env=real_env,
                                         leo_agent_dict=realworld_agent_dict)
    digitalworld_trainer = OffPolicyTrainer(args=args,
                                            log=log,
                                            tb_writer=tb_writer,
                                            env=digital_env,
                                            leo_agent_dict=digitalworld_agent_dict)
    if args.running_mode == 'training':
      while digitalworld_trainer.total_eps < args.ep_max_timesteps:
        training_process(realworld_trainer, digitalworld_trainer)

      digitalworld_trainer.print_time()
      realworld_trainer.print_time()

      for agent_name, agent in realworld_agent_dict.items():
        agent.policy.save(
          filename=f'{filename}_real_world_{agent_name}', directory=saving_directory)
      for agent_name, agent in digitalworld_agent_dict.items():
        agent.policy.save(
            filename=f'{filename}_digital_world_{agent_name}', directory=saving_directory)
      misc.save_config(args=args)

    elif args.running_mode == 'testing':
      print(f'Testing {filename}......')
      for agent_name, agent in realworld_agent_dict.items():
        agent.policy.load(
          filename=f'{filename}_real_world_{agent_name}', directory=loading_directory)
      for agent_name, agent in digitalworld_agent_dict.items():
        agent.policy.load(
          filename=f'{filename}_digital_world_{agent_name}', directory=loading_directory)

      testing_process()
    else:
      raise ValueError(f'No {args.running_mode} running mode')

  else:
    raise ValueError('On-policy trainer dict is not yet finished.')

  tb_writer.close()


def training_process(realworld_trainer: OffPolicyTrainer, digitalworld_trainer: OffPolicyTrainer):
  # run one episode with epsilon greedy
  realworld_trainer.collect_one_episode()
  digitalworld_trainer.collect_one_episode()

  # train the neural network
  realworld_trainer.train()
  digitalworld_trainer.train()

  # real world digital twin joint RA

  # evaluate performance
  realworld_trainer.eval_progress()
  digitalworld_trainer.eval_progress()


def testing_process(realworld_trainer: OffPolicyTrainer, digitalworld_trainer: OffPolicyTrainer):
  realworld_trainer.eval_progress(str='testing')
  digitalworld_trainer.eval_progress(str='testing')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')

  # NN Training
  parser.add_argument(
    '--model', default='TD3', type=str,
    help='Learnig model')
  parser.add_argument(
    '--batch-size', default=16, type=int,
    help='Batch size for both actor and critic')
  parser.add_argument(
      '--actor-lr', default=1e-6, type=float,
      help='Learning rate for actor')
  parser.add_argument(
      '--critic-lr', default=2e-5, type=float,
      help='Learning rate for critic')
  parser.add_argument(
      '--lr-reduce-factor', default=0.999, type=float,
      help='Reduce factor of learning rate')
  parser.add_argument(
      '--lr-reduce-patience', default=5000, type=int,
      help='Patience of reducing learning rate')
  parser.add_argument(
      '--lambda-l2', default=1e-9, type=float,
      help='L2 regularization factor')
  parser.add_argument(
      '--clipping-grad-norm', default=1, type=float,
      help='Value of clipping grad norm')
  parser.add_argument(
      '--ra-actor-n-hidden', default=3200, type=int,
      help='Number of hidden neuron')
  parser.add_argument(
      '--ra-critic-n-hidden', default=6400, type=int,
      help='Number of hidden neuron')
  parser.add_argument(
      '--ch-actor-n-hidden', default=1200, type=int,
      help='Number of hidden neuron')
  parser.add_argument(
      '--ch-critic-n-hidden', default=2400, type=int,
      help='Number of hidden neuron')
  parser.add_argument(
      '--iter-num', default=4, type=int,
      help='Number of base training iteration')
  parser.add_argument(
      '--replay-buffer-size', default=10000, type=int,
      help='The printing number of the network weight (for debug)')

  # --------------- TD3 -----------------------
  parser.add_argument(
      '--tau', default=0.01, type=float,
      help='Target network update rate')
  parser.add_argument(
      '--policy-freq', default=2, type=int,
      help='Frequency of delayed policy updates')
  parser.add_argument(
      '--min-epsilon', default=0.35, type=float,
      help='The minimum of epsilon')
  parser.add_argument(
      '--expl-noise', default=0.2, type=float,
      help='The stdv of the exploration noise')
  parser.add_argument(
      '--policy-noise', default=0.1, type=float,
      help='The policy noise')
  parser.add_argument(
      '--noise-clip', default=0.1, type=float,
      help='The clip range of policy noise')
  parser.add_argument(
      '--epsilon-decay-rate', default=0.99999, type=float,
      help='The rate of epsilon decay')
  parser.add_argument(
      '--discount', default=1e-2, type=float,
      help='Discount factor')
  parser.add_argument(
      '--full-explore-steps', default=1e4, type=int,
      help='Number of steps to do exploration')

  # -----------Parameter Sharing ---------------
  parser.add_argument(
      '--federated-update-rate', default=5e-3, type=float,
      help='Network exchanging rate of federated agents')
  parser.add_argument(
      '--federated-upload-freq', default=5, type=int,
      help='Frequency of federated uploading per learning iteration')
  parser.add_argument(
      '--federated-download-freq', default=15, type=int,
      help='Frequency of federated downloading per learning iteration')
  parser.add_argument(
      '--historical-smoothing-coef', default=0.9, type=float,
      help='The smoothing coefficient of the historical average reward')
  parser.add_argument(
      '--max-sharing-weight', default=2, type=float,
      help='Maximum weight of parameter sharing for each agent')
  parser.add_argument(
      '--uploaded-layer-num-per-turn', default=2, type=int,
      help='number of layers per uploading')
  parser.add_argument(
      '--partial-upload-type', default='by-turns', type=str,
      help='"random" or "by-turns"')

  # ---------------- A3C baseline -------------------
  parser.add_argument(
      '--a3c-global-update-freq', default=5, type=int,
      help='Frequency of global updating in A3C')

  # ------------------- Env -------------------------
  parser.add_argument(
      '--real-env-name', type=str, required=True,
      help='OpenAI gym environment name. Correspond to the real world')
  parser.add_argument(
      '--digital-env-name', type=str, required=True,
      help='OpenAI gym environment name. Correspond to the digital twins')
  parser.add_argument(
      '--ep-max-timesteps', type=int, required=True,
      help='Total number of episodes')
  parser.add_argument(
      '--step-per-ep', default=50, type=int,
      help='Total number of steps in one episode')
  parser.add_argument(
      '--eval-period', default=10, type=int,
      help='The evaluation frequency')

  # ------------------ Misc -------------------------
  parser.add_argument(
      '--print-weight-num', default=10, type=int,
      help='The printing number of the network weight (for debug)')
  parser.add_argument(
      '--prefix', default='', type=str,
      help='Prefix for tb_writer and logging')
  parser.add_argument(
      '--seed', default=456789, type=int,
      help='Sets Gym, PyTorch and Numpy seeds')
  parser.add_argument(
      '--running-mode', default='training', type=str,
      help='Training or Testing')

  args = parser.parse_args()

  # Set log name

  now = datetime.now()
  time_string = now.strftime('%Y_%m_%d_%H_%M_%S')
  args.log_name = f'{args.env_name}_{args.prefix}_log_{time_string}'

  main(args=args)
