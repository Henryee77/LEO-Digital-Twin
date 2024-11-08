import os


if __name__ == '__main__':
  ue_num_list = [3]
  max_ep = 2_000
  mode = 'DT FULL'

  if mode == 'DT FULL':
    d_start_ep = 0
    r_start_ep = round(max_ep / 2)
  elif mode == 'Real Only':
    d_start_ep = max_ep + 1
    r_start_ep = 0
  else:
    raise ValueError(f'No such {mode} system architecture.')

  prefix = f'{mode} {max_ep} eps'

  for ue_num in ue_num_list:
    error_code = os.system(
      f'python main.py --model "TD3" --max-ep-num {max_ep} '
      f'--dt_online_ep {d_start_ep} '
      f'--realLEO_online_ep {r_start_ep} --ue-num {ue_num} --prefix "{prefix}"'
    )
    if error_code > 0:
      print('--------------------------------------------------------------------------------------------------')
      raise ValueError('Runtime error.')
