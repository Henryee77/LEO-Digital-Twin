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
  tb_writer = SummaryWriter(logdir=f'./log/tb_{args.log_name}', flush_secs=60)
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

  sat_name_list = ['3_0_24', '2_0_1', '1_0_9']
  # Initialize agents
  realworld_agent_dict = {}
  digitalworld_agent_dict = {}
  for sat_name in sat_name_list:
    realworld_agent_dict[sat_name] = Agent(policy_name=args.model,
                                           tb_writer=tb_writer,
                                           log=log,
                                           sat_name=sat_name,
                                           agent_type='real_LEO',
                                           args=args,
                                           device=device)
    digitalworld_agent_dict[sat_name] = Agent(policy_name=args.model,
                                              tb_writer=tb_writer,
                                              log=log,
                                              sat_name=sat_name,
                                              agent_type='digital_LEO',
                                              args=args,
                                              device=device)

  # Create env
  real_env = misc.make_env(args.real_env_name,
                           args=args,
                           ax=ax,
                           tb_writer=tb_writer,
                           agent_dict=realworld_agent_dict,
                           real_agents=realworld_agent_dict,
                           digital_agents=digitalworld_agent_dict,
                           agent_names=sat_name_list)
  digital_env = misc.make_env(args.digital_env_name,
                              args=args,
                              ax=ax,
                              tb_writer=tb_writer,
                              agent_dict=digitalworld_agent_dict,
                              real_agents=realworld_agent_dict,
                              digital_agents=digitalworld_agent_dict,
                              agent_names=sat_name_list)
  print(f'real env name: {real_env.name}')
  print(f'digital env name: {digital_env.name}')

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
    realworld_trainer.set_twin_trainer(digitalworld_trainer)
    digitalworld_trainer.set_twin_trainer(realworld_trainer)

    if args.running_mode == 'training':
      # Turn off the real world trainer, only digital twins are doing pretraining
      realworld_trainer.online = False

      while digitalworld_trainer.total_eps < args.ep_max_timesteps:
        training_process(args, realworld_trainer, digitalworld_trainer)
        if digitalworld_trainer.total_eps == args.pretraining_eps:
          realworld_trainer.copy_NN_from_twin()
          realworld_trainer.online = True
          realworld_trainer.total_eps = digitalworld_trainer.total_eps

          # evaluate performance every certain steps
        if digitalworld_trainer.total_eps % args.eval_period == 0:
          eval_process(args, realworld_trainer, digitalworld_trainer)

      digitalworld_trainer.print_time()

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

      eval_process(args, realworld_trainer, digitalworld_trainer, running_mode=args.running_mode)
    else:
      raise ValueError(f'No \'{args.running_mode}\' running mode')

  else:
    raise ValueError('On-policy trainer dict is not yet finished.')

  tb_writer.close()


def training_process(args, realworld_trainer: OffPolicyTrainer, digitalworld_trainer: OffPolicyTrainer):
  # run one episode with epsilon greedy
  run_one_eps(args=args,
              realworld_trainer=realworld_trainer,
              digitalworld_trainer=digitalworld_trainer)

  # train the neural network
  digitalworld_trainer.train()
  realworld_trainer.train()


def eval_process(args, realworld_trainer: OffPolicyTrainer, digitalworld_trainer: OffPolicyTrainer, running_mode='training'):
  step_count = 0
  digital_done = real_done = False
  digitalworld_trainer.reset_env()
  realworld_trainer.reset_env()

  while step_count < args.max_step_per_ep and not (digital_done or real_done):
    digital_actions = digitalworld_trainer.deterministic_actions()
    real_actions = realworld_trainer.deterministic_actions()

    _, _, _, digital_done = digitalworld_trainer.take_action(digital_actions, running_mode=running_mode)
    _, _, _, real_done = realworld_trainer.take_action(real_actions, running_mode=running_mode)

    step_count += 1

  digitalworld_trainer.save_eval_result(step_count)
  realworld_trainer.save_eval_result(step_count)


def run_one_eps(args, realworld_trainer: OffPolicyTrainer, digitalworld_trainer: OffPolicyTrainer):
  step_count = 0
  digital_done = real_done = False
  digitalworld_trainer.reset_env()
  realworld_trainer.reset_env()

  while step_count < args.max_step_per_ep and not (digital_done or real_done):
    digital_actions = digitalworld_trainer.stochastic_actions()
    real_actions = realworld_trainer.stochastic_actions()

    (digital_prev_state_dict,
     digital_action_dict,
     digital_step_total_reward,
     digital_done) = digitalworld_trainer.take_action(digital_actions)
    (real_prev_state_dict,
     real_action_dict,
     real_step_total_reward,
     real_done) = realworld_trainer.take_action(real_actions)

    digitalworld_trainer.save_to_replaybuffer(prev_state_dict=digital_prev_state_dict,
                                              action_dict=digital_action_dict,
                                              total_reward=digital_step_total_reward,
                                              done=digital_done)
    realworld_trainer.save_to_replaybuffer(prev_state_dict=real_prev_state_dict,
                                           action_dict=real_action_dict,
                                           total_reward=real_step_total_reward,
                                           done=real_done)

    step_count += 1

  digitalworld_trainer.save_training_result(step_count)
  realworld_trainer.save_training_result(step_count)


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
      '--actor-n-hidden', default=4200, type=int,
      help='Number of hidden neuron')
  parser.add_argument(
      '--critic-n-hidden', default=8400, type=int,
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
      '--real-env-name', type=str, default='RealWorld-v0',
      help='OpenAI gym environment name. Correspond to the real world')
  parser.add_argument(
      '--digital-env-name', type=str, default='DigitalWorld-v0',
      help='OpenAI gym environment name. Correspond to the digital twins')
  parser.add_argument(
      '--ep-max-timesteps', type=int, required=True,
      help='Total number of episodes')
  parser.add_argument(
      '--max-step-per-ep', default=50, type=int,
      help='Total number of steps in one episode')
  parser.add_argument(
      '--eval-period', default=10, type=int,
      help='The evaluation frequency')
  parser.add_argument(
      '--pretraining-eps', default=7000, type=int,
      help='The number of episodes for pretraining digital twins')

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
  args.log_name = f'{args.prefix}_log_{time_string}'

  main(args=args)
