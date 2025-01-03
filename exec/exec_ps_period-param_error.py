import sys
import subprocess
from low_earth_orbit.util import constant


def mode_2_start_ep(mode):
  if mode == 'DT + TS':
    d_start_ep = 0
    r_start_ep = 0
    fs_period = 80
    twin_sharing_period = 5
  elif mode == 'DT':
    d_start_ep = 0
    r_start_ep = 0
    fs_period = (max_ep + 1) * step_num
    twin_sharing_period = (max_ep + 1) * step_num
  elif mode == 'No DT':
    d_start_ep = max_ep + 1
    r_start_ep = 0
    fs_period = 80
    twin_sharing_period = (max_ep + 1) * step_num
  else:
    raise ValueError(f'No such {mode} system architecture.')
  return d_start_ep, r_start_ep, fs_period, twin_sharing_period


if __name__ == '__main__':
  ue_num_list = [3]
  max_ep = 500
  step_num = 100
  mode = 'DT + TS'
  d_start_ep, r_start_ep, fs_period, _ = mode_2_start_ep(mode)
  tf = constant.DEFAULT_ACTION_TIMESLOT
  bs_mode = 'ABS'
  f_comp = 4e9
  param_error_list = [0.05 * i for i in range(0, 6)]
  param_sharing_period_list = [20 * i for i in range(1, 6)]

  dir_name = f'env_ps_period-param_error {max_ep} eps'
  # dir_name = 'debug'

  for ue_num in ue_num_list:
    for env_ps_period in param_sharing_period_list:
      for param_error in param_error_list:
        prefix = f'env_ps_period-{env_ps_period} param_error-{round(param_error * 100)}% ue{ue_num}'

        cmd = (
          f'main.py --model TD3 --max-ep-num {max_ep} --max-time-per-ep {step_num} '
          f'--action-timeslot {tf} '
          f'--dt-param-error {param_error} '
          f'--env-param-sharing-period {env_ps_period} '
          f'--beam-sweeping-mode {bs_mode} '
          f'--dt-computaion-speed {f_comp} '
          f'--dt-online-ep {d_start_ep} --realLEO-online-ep {r_start_ep} '
          f'--federated-upload-period {fs_period} --federated-download-period {fs_period} '
          f'--ue-num {ue_num}'
        )
        path_cmd = ['--prefix', prefix, '--dir-name', dir_name]
        call_cmd = cmd.split(' ')
        proc = subprocess.call([sys.executable] + call_cmd + path_cmd)
