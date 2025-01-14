"""main.py"""
import os
import argparse
import time
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from low_earth_orbit.util import constant
from MA_TD3.misc import misc
from MA_TD3.agent import Agent
from MA_TD3.trainer import OffPolicyTrainer


def main(args):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(device)
  # Create directories
  tb_path = f'./tb_result/{args.dir_name}'
  log_path = f'./log/{args.dir_name}'
  if not os.path.exists(tb_path):
    os.makedirs(tb_path)
  if not os.path.exists(log_path):
    os.makedirs(log_path)
  if not os.path.exists('./pytorch_models'):
    os.makedirs('./pytorch_models')
  if not os.path.exists('./config'):
    os.makedirs('./config')

  # Set logs
  tb_writer = SummaryWriter(log_dir=f'{tb_path}/tb_{args.log_name}')
  log = misc.set_log(args, log_path=log_path)
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
                                           device=device,
                                           comp_freq=constant.DEFAULT_LEO_CPU_CYCLE,
                                           total_agent_num=len(sat_name_list))
    digitalworld_agent_dict[sat_name] = Agent(policy_name=args.model,
                                              tb_writer=tb_writer,
                                              log=log,
                                              sat_name=sat_name,
                                              agent_type='digital_LEO',
                                              args=args,
                                              device=device,
                                              comp_freq=args.dt_computaion_speed,
                                              total_agent_num=len(sat_name_list))

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
  print(f'real env name: {real_env.unwrapped.name},'
        f'{real_env.unwrapped.leo_agents is real_env.unwrapped.real_agents}')
  print(f'digital env name: {digital_env.unwrapped.name},'
        f'{digital_env.unwrapped.leo_agents is digital_env.unwrapped.digital_agents}')

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
    realworld_trainer.twin_trainer = digitalworld_trainer
    digitalworld_trainer.twin_trainer = realworld_trainer

    if args.running_mode == 'training':
      for ep_cnt in range(1, args.max_ep_num + 1):
        if max(digitalworld_trainer.total_eps, realworld_trainer.total_eps) == args.dt_online_ep:
          digitalworld_trainer.online = True
        if max(digitalworld_trainer.total_eps, realworld_trainer.total_eps) == args.realLEO_online_ep:
          realworld_trainer.online = True

        training_process(args, realworld_trainer, digitalworld_trainer)

        # evaluate performance every certain steps
        if ep_cnt % args.eval_period == 0 or ep_cnt == 1:
          eval_process(args, realworld_trainer, digitalworld_trainer)
          tb_writer.flush()

      digitalworld_trainer.print_time()
      realworld_trainer.print_time()

      for _, agent in realworld_agent_dict.items():
        agent.policy.save(
          filename=f'{filename}_{agent.name}', directory=saving_directory)
      for _, agent in digitalworld_agent_dict.items():
        agent.policy.save(
            filename=f'{filename}_{agent.name}', directory=saving_directory)
      misc.save_config(args=args)

    elif args.running_mode == 'testing':
      print(f'Testing {filename}......')
      for _, agent in realworld_agent_dict.items():
        agent.policy.load(
          filename=f'{filename}_{agent.name}', directory=loading_directory)
      for _, agent in digitalworld_agent_dict.items():
        agent.policy.load(
          filename=f'{filename}_{agent.name}', directory=loading_directory)

      eval_process(args, realworld_trainer, digitalworld_trainer, running_mode=args.running_mode)
    else:
      raise ValueError(f'No \'{args.running_mode}\' running mode')

  else:
    raise ValueError('On-policy trainer dict is not yet finished.')

  tb_writer.close()


def eval_process(args, realworld_trainer: OffPolicyTrainer, digitalworld_trainer: OffPolicyTrainer, running_mode='training'):
  time_count = 0
  digital_done = real_done = False
  d_info = digitalworld_trainer.reset_env(eval=True)
  r_info = realworld_trainer.reset_env(eval=True)

  while time_count < args.max_time_per_ep and not (digital_done or real_done):
    if (realworld_trainer.online and r_info['has_action']) or (digitalworld_trainer.online and d_info['has_action']):
      digital_actions = digitalworld_trainer.deterministic_actions()
      real_actions = realworld_trainer.deterministic_actions()

      _, _, digital_done, d_info = digitalworld_trainer.take_action(digital_actions, running_mode=running_mode)
      _, _, real_done, r_info = realworld_trainer.take_action(real_actions, running_mode=running_mode)

    else:
      _, digital_done, d_info = digitalworld_trainer.no_action_step()
      _, real_done, r_info = realworld_trainer.no_action_step()

    time_count += 1

  digitalworld_trainer.save_eval_result(time_count)
  realworld_trainer.save_eval_result(time_count)


def training_process(args, realworld_trainer: OffPolicyTrainer, digitalworld_trainer: OffPolicyTrainer):
  time_count = 0
  digital_done = real_done = False
  d_info = digitalworld_trainer.reset_env()
  r_info = realworld_trainer.reset_env()

  while time_count < args.max_time_per_ep and not (digital_done or real_done):
    if (realworld_trainer.online and r_info['has_action']) or (digitalworld_trainer.online and d_info['has_action']):
      digital_actions = digitalworld_trainer.stochastic_actions()
      real_actions = realworld_trainer.stochastic_actions()

      (digital_prev_state_dict,
       digital_step_reward_dict,
       digital_done,
       d_info) = digitalworld_trainer.take_action(digital_actions)
      (real_prev_state_dict,
       real_step_reward_dict,
       real_done,
       r_info) = realworld_trainer.take_action(real_actions)

      if time_count % args.env_param_sharing_period == 0:
        digitalworld_trainer.receive_sensing_env_param()

      if time_count % args.model_sharing_period == 0:
        digitalworld_trainer.twin_parameter_query()
        realworld_trainer.twin_parameter_query()
        digitalworld_trainer.twin_parameter_update()
        realworld_trainer.twin_parameter_update()

      digitalworld_trainer.save_to_replaybuffer(agent_type='DT',
                                                prev_state_dict=digital_prev_state_dict,
                                                action_dict=digital_actions,
                                                reward_dict=digital_step_reward_dict,
                                                done=digital_done)
      realworld_trainer.save_to_replaybuffer(agent_type='LEO',
                                             prev_state_dict=real_prev_state_dict,
                                             action_dict=real_actions,
                                             reward_dict=real_step_reward_dict,
                                             done=real_done)
    else:
      _, digital_done, d_info = digitalworld_trainer.no_action_step()
      _, real_done, r_info = realworld_trainer.no_action_step()

    time_count += 1

  digitalworld_trainer.save_training_result(time_count)
  realworld_trainer.save_training_result(time_count)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')

  # ----------------- NN --------------------
  parser.add_argument(
    '--model', default='TD3', type=str,
    help='Learnig model')
  parser.add_argument(
    '--batch-size', default=16, type=int,
    help='Batch size for both actor and critic')
  parser.add_argument(
      '--actor-lr', default=5e-5, type=float,
      help='Learning rate for actor')
  parser.add_argument(
      '--critic-lr', default=1e-4, type=float,
      help='Learning rate for critic')
  parser.add_argument(
      '--lr-reduce-factor', default=0.9, type=float,
      help='Reduce factor of learning rate')
  parser.add_argument(
      '--lr-reduce-patience', default=100, type=int,
      help='Patience of reducing learning rate')
  parser.add_argument(
      '--lambda-l2', default=1e-9, type=float,
      help='L2 regularization factor')
  parser.add_argument(
      '--clipping-grad-norm', default=1, type=float,
      help='Value of clipping grad norm')
  parser.add_argument(
      '--actor-n-hidden', default=2500, type=int,
      help='Number of hidden neuron')
  parser.add_argument(
      '--critic-n-hidden', default=5000, type=int,
      help='Number of hidden neuron')
  parser.add_argument(
      '--training-period', default=50, type=int,
      help='Peiord (number of radio frame) of NN training.')
  parser.add_argument(
      '--replay-buffer-size', default=2000, type=int,
      help='The size of replay buffer')

  # --------------- TD3 -----------------------
  parser.add_argument(
      '--tau', default=0.01, type=float,
      help='Target network update rate')
  parser.add_argument(
      '--policy-freq', default=2, type=int,
      help='Frequency of delayed policy updates')
  parser.add_argument(
      '--min-epsilon', default=0, type=float,
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
      '--epsilon-decay-rate', default=0.9999, type=float,
      help='The rate of epsilon decay')
  parser.add_argument(
      '--discount', default=1e-2, type=float,
      help='Discount factor')
  parser.add_argument(
      '--full-explore-steps', default=1e3, type=int,
      help='Number of steps to do exploration')

  # -----------Federated Sharing ---------------
  parser.add_argument(
      '--federated-update-rate', default=1e-1, type=float,
      help='Network exchanging rate of federated agents')
  parser.add_argument(
      '--federated-upload-period', default=20, type=int,
      help='Period of federated uploading')
  parser.add_argument(
      '--federated-download-period', default=20, type=int,
      help='Period of federated downloading')
  parser.add_argument(
      '--federated-layer-num-per-turn', default=1, type=int,
      help='number of layers per federated uploading')
  parser.add_argument(
      '--partial-upload-type', default='by-turns', type=str,
      help='"random" or "by-turns"')

  # ------------Twin Sharing ---------------
  parser.add_argument(
      '--twin-sharing-update-rate', default=0.5, type=float,
      help='Network update rate of twin sharing')
  parser.add_argument(
      '--model-sharing-period', default=5, type=int,
      help='Period of twin sharing uploading')
  parser.add_argument(
      '--env-param-sharing-period', default=5, type=int,
      help='Period of twin sharing uploading')
  parser.add_argument(
      '--twin-sharing-layer-num-per-turn', default=1, type=int,
      help='number of layers per twin sharing')
  parser.add_argument(
      '--historical-reward-window', default=100, type=int,
      help='Window size of calculating the historical average reward')
  parser.add_argument(
      '--sharing-weight-growth-rate', default=2, type=float,
      help='The growth rate of the generalised logistic function')
  parser.add_argument(
      '--sharing-weight-asymptote-occurrence', default=3, type=float,
      help='The nu variable of the generalised logistic function')

  # ---------------- A3C baseline -------------------
  parser.add_argument(
      '--a3c-global-update-freq', default=5, type=int,
      help='Frequency of global updating in A3C')

  # ------------------- LEO -------------------------
  parser.add_argument(
      '--beam-sweeping-mode', type=str, default='ABS',
      help='Beam-sweeping mode')
  parser.add_argument(
      '--cell-layer-num', type=int, default=constant.DEFAULT_CELL_LAYER,
      help='Number of cell layer (related to cell number)')
  parser.add_argument(
      '--leo-computaion-speed', type=float, default=constant.DEFAULT_LEO_CPU_CYCLE,
      help='LEO computation speed')

  # --------------------- DT ------------------------
  parser.add_argument(
      '--dt-computaion-speed', type=float, default=constant.DEFAULT_DT_CPU_CYCLE,
      help='DT computation speed')
  parser.add_argument(
      '--shared-state-type', type=int, default=2,
      help='Number of the state type shared. 0: None, 1: pathloss, 2: pathloss + CSI')
  parser.add_argument(
      '--dt-param-error', type=float, default=0,
      help='The error of the environment parameters sending from the real LEO')
  parser.add_argument(
      '--rt-ray-spacing', type=str, default='0.25',
      help='The spacing degree of each ray in ray tracing')

  # --------------------- UE ------------------------
  parser.add_argument(
      '--ue-num', default=3, type=int,
      help='The number of ues')
  parser.add_argument(
      '--R-min', default=1e8, type=float,
      help='QoS constraint')

  # --------------------- Agent ------------------------
  parser.add_argument(
      '--dt-online-ep', default=0, type=int,
      help='The episode to turn on digital twins')
  parser.add_argument(
      '--realLEO-online-ep', default=0, type=int,
      help='The episode to turn on real LEOs')
  parser.add_argument(
      '--scope-of-states', default='local', type=str,
      help='(Only for benchmark!) Using local observed states, or aggregate the states to global state. (input: local or global)')
  parser.add_argument(
      '--scope-of-actions', default='distributed', type=str,
      help='(Only for benchmark!) Single centralized agent, or distributed agents. (input: centralized or distributed)')

  # ------------------- Env -------------------------
  parser.add_argument(
      '--real-env-name', type=str, default='RealWorld-v0',
      help='OpenAI gym environment name. Correspond to the real world')
  parser.add_argument(
      '--digital-env-name', type=str, default='DigitalWorld-v0',
      help='OpenAI gym environment name. Correspond to the digital twins')
  parser.add_argument(
      '--max-ep-num', type=int, required=True,
      help='Total number of episodes')
  parser.add_argument(
      '--max-time-per-ep', default=100, type=int,
      help='Total number of steps in one episode')
  parser.add_argument(
      '--action-timeslot', default=constant.DEFAULT_ACTION_TIMESLOT, type=int,
      help='Total number of steps in one episode')
  parser.add_argument(
      '--eval-period', default=10, type=int,
      help='The evaluation frequency')
  parser.add_argument(
      '--has-weather-module', default=1, type=int,
      help='Has weather simulation module or not. (0=False, 1=True)')
  parser.add_argument(
      '--rainfall-prob', default=0.05, type=float,
      help='The rainfall propability of the region')

  # ------------------ Misc -------------------------
  parser.add_argument(
      '--print-weight-num', default=10, type=int,
      help='The printing number of the network weight (for debug)')
  parser.add_argument(
      '--prefix', default='', type=str,
      help='Prefix for tb_writer and logging')
  parser.add_argument(
      '--dir-name', default='', type=str,
      help='Name of the tb directory')
  parser.add_argument(
      '--seed', default=778899, type=int,
      help='Sets Gym, PyTorch and Numpy seeds')
  parser.add_argument(
      '--running-mode', default='training', type=str,
      help='Training or Testing')

  args = parser.parse_args()

  # Set log name
  time_string = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
  args.log_name = f'{args.prefix}_log_{time_string}'

  start_time = time.time()

  main(args=args)

  print(f'Total time from main: {(time.time() - start_time) / 3600: .2f} hr')
