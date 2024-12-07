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
    ps_period = 80
    twin_sharing_period = (max_ep + 1) * step_num
  else:
    raise ValueError(f'No such {mode} system architecture.')
  return d_start_ep, r_start_ep, ps_period, twin_sharing_period


if __name__ == '__main__':
  ue_num_list = [6]
  max_ep = 500
  step_num = 100
  mode = 'DT + TS'
  d_start_ep, r_start_ep, ps_period, twin_sharing_period = mode_2_start_ep(mode)
  tf = constant.DEFAULT_ACTION_TIMESLOT
  bs_mode = 'ABS'
  f_comp = 4e9
  provide_type = [0, 1, 2]
  param_error_list = [0.05 * i for i in range(0, 6)]

  dir_name = f'statetype-param_error {max_ep} eps - 2'
  # dir_name = 'debug'

  for ue_num in ue_num_list:
    for shared_state_type in provide_type:
      for param_error in param_error_list:
        prefix = f'shared_type-{shared_state_type} param_error-{param_error * 100}% ue{ue_num}'

        cmd = (
          f'main.py --model TD3 --max-ep-num {max_ep} --max-time-per-ep {step_num} '
          f'--shared-state-type {shared_state_type} '
          f'--action-timeslot {tf} '
          f'--dt-param-error {param_error} '
          f'--beam-sweeping-mode {bs_mode} '
          f'--dt-computaion-speed {f_comp} '
          f'--dt_online_ep {d_start_ep} --realLEO_online_ep {r_start_ep} '
          f'--federated-upload-period {ps_period} --federated-download-period {ps_period} '
          f'--ue-num {ue_num}'
        )
        path_cmd = ['--prefix', prefix, '--dir-name', dir_name]
        call_cmd = cmd.split(' ')
        proc = subprocess.call([sys.executable] + call_cmd + path_cmd)
