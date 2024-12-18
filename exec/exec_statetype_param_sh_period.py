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
  ue_num_list = [3]
  max_ep = 500
  step_num = 100
  mode = 'DT + TS'
  tf = 3
  bs_mode = 'ABS'
  cell_layer = 3
  f_comp = 5e9
  provide_type = [0, 1, 2]
  ts_percent = [0.1 * i for i in range(1, 6)]
  twin_sharing_period_list = [round(1 / percent) for percent in ts_percent]
  d_start_ep, r_start_ep, ps_period, _ = mode_2_start_ep(mode)

  dir_name = f'1 - statetype_param_sharing_period {max_ep} eps'
  while os.path.exists(f'./tb_result/{dir_name}'):
    print('Warning: Directory exists')
    split_str = dir_name.split('-')
    dir_name = f'{int(split_str[0]) + 1} -' + split_str[-1]

  for ue_num in ue_num_list:
    for shared_state_type in provide_type:
      for ts_period in twin_sharing_period_list:
        prefix = f'shared_type-{shared_state_type} param_sharing_period-{ts_period} ue{ue_num}'

        cmd = (
          f'main.py --model TD3 --max-ep-num {max_ep} --max-time-per-ep {step_num} '
          f'--shared-state-type {shared_state_type} '
          f'--model-sharing-period {ts_period} '
          f'--env-param-sharing-period {ts_period} '
          f'--action-timeslot {tf} '
          f'--beam-sweeping-mode {bs_mode} '
          f'--cell-layer-num {cell_layer} '
          f'--dt-computaion-speed {f_comp} '
          f'--dt-online-ep {d_start_ep} --realLEO-online-ep {r_start_ep} '
          f'--federated-upload-period {ps_period} --federated-download-period {ps_period} '
          f'--ue-num {ue_num}'
        )
        path_cmd = ['--prefix', prefix, '--dir-name', dir_name]
        call_cmd = cmd.split(' ')
        proc = subprocess.call([sys.executable] + call_cmd + path_cmd)
