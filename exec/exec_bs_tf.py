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
  ue_num_list = [6]
  max_ep = 300
  mode = 'No DT'
  cell_layer = 3
  f_comp = constant.DEFAULT_DT_CPU_CYCLE
  bs_list = ['ABS', 'SCBS', 'SSBS']
  tf_list = [1, 2, 3, 4, 5]
  step_num = 100

  dir_name = f'1 - Beam sweeping-T_f {max_ep} eps'
  while os.path.exists(f'./tb_result/{dir_name}'):
    split_str = dir_name.split('-')
    dir_name = f'{int(split_str[0]) + 1} - ' + split_str[-1]

  for ue_num in ue_num_list:
    for bs_mode in bs_list:
      for tf in tf_list:
        prefix = f'{mode} {bs_mode} tf-{tf}s ue{ue_num}'
        d_start_ep, r_start_ep, ps_period, twin_sharing_period = mode_2_start_ep(mode)

        error_code = os.system(
          f'python main.py --model "TD3" --max-ep-num {max_ep} --max-time-per-ep {step_num} '
          f'--action-timeslot {tf} '
          f'--beam-sweeping-mode {bs_mode} '
          f'--cell-layer-num {cell_layer} '
          f'--dt-computaion-speed {f_comp} '
          f'--dt-online-ep {d_start_ep} --realLEO-online-ep {r_start_ep} '
          f'--federated-upload-period {ps_period} --federated-download-period {ps_period} '
          f'--model-sharing-period {twin_sharing_period} '
          f'--ue-num {ue_num} --prefix "{prefix}" --dir-name "{dir_name}"'
        )
        if error_code > 0:
          print('--------------------------------------------------------------------------------------------------')
          raise ValueError('Runtime error.')
