import os
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
  ue_num_list = [3, 6, 9]
  max_ep = 1_000
  step_num = 100
  mode = 'DT + TS'
  tf = constant.DEFAULT_ACTION_TIMESLOT
  bs_mode = 'ABS'
  cell_layer = 3
  f_comp = 2.5e9
  provide_type = [0, 1, 2]
  twin_sharing_period_list = [5 * i for i in range(1, 10)]
  d_start_ep, r_start_ep, ps_period, twin_sharing_period = mode_2_start_ep(mode)

  dir_name = f'provided_data-twin_sharing {max_ep} eps'
  # dir_name = 'debug'
  bugged_folders = []

  for ue_num in ue_num_list:
    for shared_state_type in provide_type:
      for ts_period in twin_sharing_period_list:
        prefix = f'shared_type-{shared_state_type} ts_period-{ts_period} ue{ue_num}'

        error_code = os.system(
          f'python main.py --model "TD3" --max-ep-num {max_ep} --max-time-per-ep {step_num} '
          f'--shared-state-type {shared_state_type} '
          f'--twin-sharing-period {ts_period} '
          f'--action-timeslot {tf} '
          f'--beam-sweeping-mode {bs_mode} '
          f'--cell-layer-num {cell_layer} '
          f'--dt-computaion-speed {f_comp} '
          f'--dt_online_ep {d_start_ep} --realLEO_online_ep {r_start_ep} '
          f'--federated-upload-period {ps_period} --federated-download-period {ps_period} '
          f'--twin-sharing-period {twin_sharing_period} '
          f'--ue-num {ue_num} --prefix "{prefix}" --dir-name "{dir_name}"'
        )
        if error_code > 0:
          print('--------------------------------------------------------------------------------------------------')
          # raise ValueError('Runtime error.')
          bugged_folders.append(prefix)

  print(bugged_folders)
