import sys
import os
import subprocess
from low_earth_orbit.util import constant


def mode_2_start_ep(mode):
  if mode == 'DT + TS + FS':
    d_start_ep = 0
    r_start_ep = 0
    fs_period = 20
    twin_sharing_period = 20
  elif mode == 'DT offline':
    d_start_ep = 0
    r_start_ep = round(max_ep / 2)
    fs_period = (max_ep + 1) * step_num
    twin_sharing_period = (max_ep + 1) * step_num
  elif mode == 'DT no NN':
    d_start_ep = 0
    r_start_ep = 0
    fs_period = (max_ep + 1) * step_num
    twin_sharing_period = (max_ep + 1) * step_num
  else:
    raise ValueError(f'No such {mode} system architecture.')
  return d_start_ep, r_start_ep, fs_period, twin_sharing_period


if __name__ == '__main__':
  ue_num_list = [3, 6, 9, 12, 15]
  max_ep = 500
  tf = 3
  bs_mode = 'ABS'
  cell_layer = 3
  actor_lr = 4e-5
  comp_speed = constant.DEFAULT_LEO_CPU_CYCLE * 2
  f_comp = constant.DEFAULT_DT_CPU_CYCLE
  mode_list = ['DT + TS + FS']
  step_num = 100
  env_sharing_period = 5
  state_type = 'global'

  dir_name = f'Benchmark {max_ep} eps'

  for ue_num in ue_num_list:
    for mode in mode_list:
      prefix = f'{state_type} state ue{ue_num}'
      d_start_ep, r_start_ep, fs_period, twin_sharing_period = mode_2_start_ep(mode)
      cmd = (
        f'main.py --model TD3 --max-ep-num {max_ep} --max-time-per-ep {step_num} '
        f'--action-timeslot {tf} '
        f'--actor-lr {actor_lr} '
        f'--scope-of-states {state_type} '
        f'--leo-computaion-speed {comp_speed} '
        f'--dt-online-ep {d_start_ep} --realLEO-online-ep {r_start_ep} '
        f'--federated-upload-period {fs_period} --federated-download-period {fs_period} '
        f'--model-sharing-period {twin_sharing_period} '
        f'--env-param-sharing-period {env_sharing_period} '
        f'--ue-num {ue_num}'
      )
      path_cmd = ['--prefix', prefix, '--dir-name', dir_name]
      call_cmd = cmd.split(' ')
      proc = subprocess.call([sys.executable] + call_cmd + path_cmd)