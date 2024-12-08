import sys
import subprocess
from low_earth_orbit.util import constant


def mode_2_start_ep(mode):
  if mode == 'DT + TS':
    d_start_ep = 0
    r_start_ep = 0
    ps_period = 80
    twin_sharing_period = 5
  elif mode == 'DT':
    d_start_ep = 0
    r_start_ep = 0
    ps_period = (max_ep + 1) * step_num
    twin_sharing_period = (max_ep + 1) * step_num
  elif mode == 'No DT':
    d_start_ep = max_ep + 1
    r_start_ep = 0
    ps_period = (max_ep + 1) * step_num
    twin_sharing_period = (max_ep + 1) * step_num
  else:
    raise ValueError(f'No such {mode} system architecture.')
  return d_start_ep, r_start_ep, ps_period, twin_sharing_period


if __name__ == '__main__':
  ue_num_list = [3]
  max_ep = 500
  tf = constant.DEFAULT_ACTION_TIMESLOT
  bs_mode = 'ABS'
  cell_layer = 3
  f_comp = constant.DEFAULT_DT_CPU_CYCLE
  mode = 'DT + TS'
  federated_sharing_period_list = [5 + 15 * i for i in range(1, 6)]
  step_num = 100

  dir_name = f'Convergence - Sharing {max_ep} eps'
  # dir_name = 'debug'

  for ue_num in ue_num_list:
    d_start_ep, r_start_ep, _, twin_sharing_period = mode_2_start_ep(mode)
    for fs_period in federated_sharing_period_list:
      prefix = f'{mode} ue{ue_num}'

      cmd = (
        f'main.py --model TD3 --max-ep-num {max_ep} --max-time-per-ep {step_num} '
        f'--action-timeslot {tf} '
        f'--beam-sweeping-mode {bs_mode} '
        f'--cell-layer-num {cell_layer} '
        f'--dt-computaion-speed {f_comp} '
        f'--dt_online_ep {d_start_ep} --realLEO_online_ep {r_start_ep} '
        f'--model-sharing-period {twin_sharing_period} '
        f'--env-param-sharing-period {twin_sharing_period} '
        f'--federated-upload-period {fs_period} --federated-download-period {fs_period} '
        f'--ue-num {ue_num}'
      )
      path_cmd = ['--prefix', prefix, '--dir-name', dir_name]
      call_cmd = cmd.split(' ')
      proc = subprocess.call([sys.executable] + call_cmd + path_cmd)
