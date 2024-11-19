import os


if __name__ == '__main__':
  ue_num_list = [3]
  max_ep = 1_000
  mode_list = ['DT', 'No DT']

  def mode_2_start_ep(mode):
    if mode == 'DT':
      d_start_ep = 0
      r_start_ep = 0
    elif mode == 'No DT':
      d_start_ep = max_ep + 1
      r_start_ep = 0
    else:
      raise ValueError(f'No such {mode} system architecture.')
    return d_start_ep, r_start_ep

  # dir_name = f'Baseline Comparison {max_ep} eps'
  dir_name = 'new feature'

  for ue_num in ue_num_list:
    for mode in mode_list:
      prefix = f'{mode} ue{ue_num}'
      d_start_ep, r_start_ep = mode_2_start_ep(mode)

      error_code = os.system(
        f'python main.py --model "TD3" --max-ep-num {max_ep} '
        f'--dt_online_ep {d_start_ep} '
        f'--realLEO_online_ep {r_start_ep} --ue-num {ue_num} --prefix "{prefix}" --dir-name "{dir_name}"'
      )
      if error_code > 0:
        print('--------------------------------------------------------------------------------------------------')
        raise ValueError('Runtime error.')
