import sys
import os
import subprocess
from low_earth_orbit.util import constant


def mode_2_start_ep(mode):
  if mode == 'DT + TS + FS':
    d_start_ep = 0
    r_start_ep = 0
    fs_period = 20
    twin_sharing_period = 20
  elif mode == 'DT + TS':
    d_start_ep = 0
    r_start_ep = 0
    fs_period = (max_ep + 1) * step_num
    twin_sharing_period = 20
  elif mode == 'DT':
    d_start_ep = 0
    r_start_ep = 0
    fs_period = (max_ep + 1) * step_num
    twin_sharing_period = (max_ep + 1) * step_num
  elif mode == 'No DT':
    d_start_ep = max_ep + 1
    r_start_ep = 0
    fs_period = (max_ep + 1) * step_num
    twin_sharing_period = (max_ep + 1) * step_num
  else:
    raise ValueError(f'No such {mode} system architecture.')
  return d_start_ep, r_start_ep, fs_period, twin_sharing_period


if __name__ == '__main__':
  ue_num_list = [6]
  max_ep = 500
  has_weather_list = [0, 1]
  rain_prob_list = [i * 0.1 for i in range(6)]
  mode = 'DT + TS + FS'
  step_num = 100

  d_start_ep, r_start_ep, fs_period, twin_sharing_period = mode_2_start_ep(mode)

  dir_name = f'1 - Weather Comparison {max_ep} eps'
  while os.path.exists(f'./tb_result/{dir_name}'):
    print('Warning: Directory exists')
    # raise ValueError('Directory exists')
    split_str = dir_name.split('-')
    dir_name = f'{int(split_str[0]) + 1} -' + split_str[-1]

  for ue_num in ue_num_list:
    for rain_prob in rain_prob_list:
      for has_weather in has_weather_list:
        prefix = f'{has_weather > 0} weather rain_prob {round(rain_prob, 1)} ue{ue_num}'

        cmd = (
          f'main.py --model TD3 --max-ep-num {max_ep} --max-time-per-ep {step_num} '
          f'--rainfall-prob {rain_prob} '
          f'--has-weather-module {has_weather} '
          f'--dt-online-ep {d_start_ep} --realLEO-online-ep {r_start_ep} '
          f'--federated-upload-period {fs_period} --federated-download-period {fs_period} '
          f'--model-sharing-period {twin_sharing_period} '
          f'--env-param-sharing-period {twin_sharing_period} '
          f'--ue-num {ue_num}'
        )
        path_cmd = ['--prefix', prefix, '--dir-name', dir_name]
        call_cmd = cmd.split(' ')
        proc = subprocess.call([sys.executable] + call_cmd + path_cmd)
