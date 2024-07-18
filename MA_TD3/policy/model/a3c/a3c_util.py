"""a3c_utils.py"""
from typing import Dict, List
import torch
import numpy as np
import numpy.typing as npt
from .a3c import ActorCritic


def ts_wrap(np_array, dtype=np.float32):
  if np_array.dtype != dtype:
    np_array = np_array.astype(dtype)
  return torch.from_numpy(np_array)


def push_and_pull(optimizer_dict, local_net_dict: Dict[str, ActorCritic], global_net_dict,
                  done, new_s: Dict[str, npt.NDArray], buffer_s: Dict[str, List], buffer_a: Dict[str, List], buffer_r: Dict[str, List], discount):
  for agent_name, local_net in local_net_dict.items():
    global_net = global_net_dict[agent_name]
    optimizer = optimizer_dict[agent_name]
    if done:
      new_q = 0.               # terminal
    else:
      _, new_q = local_net.forward(new_s[agent_name])

    buffer_v_target = []
    br = buffer_r[agent_name]
    for r in br[::-1]:   # iterate through reveresed list
      new_q = r + discount * new_q
      buffer_v_target.append(new_q)
    buffer_v_target.reverse()

    total_loss = 0
    for s, a, v_t, in zip(buffer_s, buffer_a, buffer_v_target):
      value_loss, policy_loss = local_net.loss_func(ts_wrap(s), ts_wrap(a), ts_wrap(v_t))
      total_loss += value_loss + policy_loss

    # calculate local gradients and push local parameters to global
    optimizer.zero_grad()
    total_loss.backward()
    for l_param, g_param in zip(local_net.parameters(), global_net.parameters()):
      g_param._grad = l_param.grad
    optimizer.step()

    # pull global parameters
    local_net.load_state_dict(global_net.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
  with global_ep.get_lock():
    global_ep.value += 1
  with global_ep_r.get_lock():
    global_ep_r.value = ep_r
    res_queue.put(global_ep_r.value)
  print(
      f'{name}'
      f'Ep: global_ep.value, '
      f'| Ep_r: {global_ep_r.value:.2f}')
