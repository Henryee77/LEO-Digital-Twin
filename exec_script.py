import os


def system_arch(s, max_ep):
  if s == 'DT FULL':
    d_start_ep = 0
    r_start_ep = round(max_ep / 2)
  elif s == 'Real Only':
    d_start_ep = max_ep + 1
    r_start_ep = 0
  else:
    raise ValueError(f'No such {s} system architecture.')

  return d_start_ep, r_start_ep


if __name__ == '__main__':
  ue_num_list = [3]
  max_ep = 5_000
  mode = 'DT FULL'
  d_start_ep, r_start_ep = system_arch(mode, max_ep)

  for ue_num in ue_num_list:
    error_code = os.system(
      f'python main.py --model "TD3" --max-ep-num {max_ep} --dt_online_ep {d_start_ep} '
      f'--realLEO_online_ep {r_start_ep} --ue-num {ue_num} --prefix "{mode} {max_ep} eps"'
    )
    if error_code > 0:
      print('--------------------------------------------------------------------------------------------------')
      raise ValueError('Runtime error.')
