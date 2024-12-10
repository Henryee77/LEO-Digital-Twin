import sys
import subprocess
from low_earth_orbit.util import constant


def mode_2_start_ep(mode):
  if mode == 'DT + TS + FS':
    d_start_ep = 0
    r_start_ep = 0
    fs_period = 20
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
  ue_num_list = [6]
  max_ep = 1000
  tf = constant.DEFAULT_ACTION_TIMESLOT
  bs_mode = 'ABS'
  cell_layer = 3
  f_comp = constant.DEFAULT_DT_CPU_CYCLE
  mode_list = ['DT + TS + FS', 'DT + TS', 'DT', 'No DT']
  step_num = 100

  dir_name = f'2 - Baseline Comparison {max_ep} eps'
  # dir_name = 'debug'

  for ue_num in ue_num_list:
    for mode in mode_list:
      prefix = f'{mode} ue{ue_num}'
      d_start_ep, r_start_ep, fs_period, twin_sharing_period = mode_2_start_ep(mode)

      cmd = (
        f'main.py --model TD3 --max-ep-num {max_ep} --max-time-per-ep {step_num} '
        f'--action-timeslot {tf} '
        f'--beam-sweeping-mode {bs_mode} '
        f'--cell-layer-num {cell_layer} '
        f'--dt-computaion-speed {f_comp} '
        f'--dt_online_ep {d_start_ep} --realLEO_online_ep {r_start_ep} '
        f'--federated-upload-period {fs_period} --federated-download-period {fs_period} '
        f'--twin-sharing-period {twin_sharing_period} '
        f'--ue-num {ue_num}'
      )
      path_cmd = ['--prefix', prefix, '--dir-name', dir_name]
      call_cmd = cmd.split(' ')
      proc = subprocess.call([sys.executable] + call_cmd + path_cmd)
