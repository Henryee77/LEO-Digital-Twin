import sys
import os
import subprocess
import time
from low_earth_orbit.util import constant


def mode_2_start_ep(mode):
  if mode == 'DT + TS + FS':
    d_start_ep = 0
    r_start_ep = 0
    fs_period = 30
    twin_sharing_period = 20
  elif mode == 'DT + TS':
    d_start_ep = 0
    r_start_ep = 0
    fs_period = (max_ep + 1) * step_num
    twin_sharing_period = 20
  elif mode == 'DT':
    d_start_ep = 0
    r_start_ep = 0
    fs_period = (max_ep + 1) * step_num
    twin_sharing_period = (max_ep + 1) * step_num
  elif mode == 'No DT':
    d_start_ep = max_ep + 1
    r_start_ep = 0
    fs_period = (max_ep + 1) * step_num
    twin_sharing_period = (max_ep + 1) * step_num
  else:
    raise ValueError(f'No such {mode} system architecture.')
  return d_start_ep, r_start_ep, fs_period, twin_sharing_period


if __name__ == '__main__':
  ue_num_list = [3]
  max_ep = 500
  f_comp = constant.DEFAULT_DT_CPU_CYCLE * 1.5
  comp_speed = constant.DEFAULT_LEO_CPU_CYCLE * 2
  actor_lr = 5e-5
  explore_step = 100
  training_period = 30
  mode = 'DT + TS + FS'
  ts_percent = [0.1 * i for i in range(1, 6)]
  twin_sharing_period_list = [round(1 / percent) for percent in ts_percent]
  step_num = 100

  dir_name = f'1 - Convergence_Sharing {max_ep} eps'
  while os.path.exists(f'./tb_result/{dir_name}'):
    split_str = dir_name.split('-')
    dir_name = f'{int(split_str[0]) + 1} -' + split_str[-1]

  for ue_num in ue_num_list:
    d_start_ep, r_start_ep, fs_period, _ = mode_2_start_ep(mode)
    for twin_sharing_period in twin_sharing_period_list:
      prefix = f'{mode} sharing_period{fs_period}  ue{ue_num}'

      cmd = (
        f'main.py --model TD3 --max-ep-num {max_ep} --max-time-per-ep {step_num} '
        f'--actor-lr {actor_lr} '
        f'--full-explore-steps {explore_step} '
        f'--training-period {training_period} '
        f'--dt-computaion-speed {f_comp} '
        f'--leo-computaion-speed {comp_speed} '
        f'--dt-online-ep {d_start_ep} --realLEO-online-ep {r_start_ep} '
        f'--model-sharing-period {twin_sharing_period} '
        f'--env-param-sharing-period {twin_sharing_period} '
        f'--federated-upload-period {fs_period} --federated-download-period {fs_period} '
        f'--ue-num {ue_num}'
      )
      path_cmd = ['--prefix', prefix, '--dir-name', dir_name]
      call_cmd = cmd.split(' ')
      proc = subprocess.call([sys.executable] + call_cmd + path_cmd)
