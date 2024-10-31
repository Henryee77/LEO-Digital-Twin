import os


def system_arch(s, max_ep):
  if s == 'DT FULL':
    d_start_ep = 0
    r_start_ep = 5000
  if s == 'Real Only':
    d_start_ep = max_ep + 1
    r_start_ep = 0
  else:
    raise ValueError('No such system architecture.')

  return d_start_ep, r_start_ep


if __name__ == '__main__':
  ue_num_list = [3]
  max_ep = 10_000

  d_start_ep, r_start_ep = system_arch('Real Only', max_ep)

  for ue_num in ue_num_list:
    error_code = os.system(
      f'python main.py --model "TD3" --max-ep-num {max_ep} --dt_online_ep {d_start_ep} --realLEO_online_ep {r_start_ep} --ue-num {ue_num} --prefix "baseline 10k eps"'
    )
    if error_code > 0:
      print('--------------------------------------------------------------------------------------------------')
      raise ValueError('Runtime error.')
