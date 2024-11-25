import os


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
  max_ep = 1_000
  mode = 'DT'
  bs_list = ['SCBS', 'SSBS', 'ABS']
  tf_list = [1, 2, 3, 4, 5]
  step_num = 100

  dir_name = f'Baseline Comparison 2 {max_ep} eps'
  # dir_name = 'debug'

  for ue_num in ue_num_list:
    for bs_mode in bs_list:
      for tf in tf_list:
        prefix = f'{bs_mode} tf-{tf}s ue{ue_num}'
        d_start_ep, r_start_ep, ps_period, twin_sharing_period = mode_2_start_ep(mode)

        error_code = os.system(
          f'python main.py --model "TD3" --max-ep-num {max_ep} --max-time-per-ep {step_num} '
          f'--action-timeslot {tf} '
          f'--beam-sweeping-mode {bs_mode} '
          f'--dt_online_ep {d_start_ep} --realLEO_online_ep {r_start_ep} '
          f'--federated-upload-period {ps_period} --federated-download-period {ps_period} '
          f'--twin-sharing-period {twin_sharing_period} '
          f'--ue-num {ue_num} --prefix "{prefix}" --dir-name "{dir_name}"'
        )
        if error_code > 0:
          print('--------------------------------------------------------------------------------------------------')
          raise ValueError('Runtime error.')
