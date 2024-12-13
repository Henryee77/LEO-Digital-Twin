import sys
import subprocess
from low_earth_orbit.util import constant


def mode_2_start_ep(mode):
  if mode == 'DT + TS':
    d_start_ep = 0
    r_start_ep = 0
    ps_period = 100
    twin_sharing_period = 5
  elif mode == 'DT':
    d_start_ep = 0
    r_start_ep = 0
    ps_period = (max_ep + 1) * step_num
    twin_sharing_period = (max_ep + 1) * step_num
  elif mode == 'No DT':
    d_start_ep = max_ep + 1
    r_start_ep = 0
    ps_period = (max_ep + 1) * step_num
    twin_sharing_period = (max_ep + 1) * step_num
  else:
    raise ValueError(f'No such {mode} system architecture.')
  return d_start_ep, r_start_ep, ps_period, twin_sharing_period


if __name__ == '__main__':
  ue_num_list = [3]
  max_ep = 500
  mode = 'DT + TS'
  cell_layer = 3
  fcomp_list = [1e9 * i for i in range(1, 10)]
  bs_mode = 'ABS'
  tf_list = [1, 2, 3, 4, 5]
  step_num = 100

  dir_name = f'T_f - f_comp {max_ep} eps - 2'
  # dir_name = 'debug'

  for ue_num in ue_num_list:
    for f_comp in fcomp_list:
      for tf in tf_list:
        prefix = f'fcomp {f_comp} tf-{tf}s ue{ue_num}'
        d_start_ep, r_start_ep, ps_period, twin_sharing_period = mode_2_start_ep(mode)

        cmd = (
            f'main.py --model TD3 --max-ep-num {max_ep} --max-time-per-ep {step_num} '
            f'--action-timeslot {tf} '
            f'--beam-sweeping-mode {bs_mode} '
            f'--cell-layer-num {cell_layer} '
            f'--dt-computaion-speed {f_comp} '
            f'--dt-online-ep {d_start_ep} --realLEO-online-ep {r_start_ep} '
            f'--federated-upload-period {ps_period} --federated-download-period {ps_period} '
            f'--twin-sharing-period {twin_sharing_period} '
            f'--ue-num {ue_num}'
        )
        path_cmd = ['--prefix', prefix, '--dir-name', dir_name]
        call_cmd = cmd.split(' ')
        proc = subprocess.call([sys.executable] + call_cmd + path_cmd)
