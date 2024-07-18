"""a3c_worker.py"""
from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING
import gymnasium as gym
import torch.multiprocessing as mp
from .a3c import ActorCritic
from misc import misc
from .a3c_util import push_and_pull, record
if TYPE_CHECKING:
  from agent import Agent
  from shared_adam import SharedAdam


class Worker(mp.Process):

  def __init__(self, state_dim, action_dim, actor_hidden_dim, critic_hidden_dim,
               min_action, max_action, global_net_dict: Dict[str, ActorCritic], optimizer_dict,
               global_ep: mp.Value, global_ep_r: mp.Value, res_queue,
               ax, device, args, name):

    super(Worker, self).__init__()
    self.name = f'w_{name}'
    print(f'initial {self.name}')
    agent_names = list(global_net_dict.keys())
    self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
    self.global_net_dict, self.optimizer_dict = global_net_dict, optimizer_dict
    self.local_net_dict = {}
    for agent_name in agent_names:
      self.local_net_dict[agent_name] = ActorCritic(state_dim=state_dim,
                                                    action_dim=action_dim,
                                                    actor_hidden_dim=actor_hidden_dim,
                                                    critic_hidden_dim=critic_hidden_dim,
                                                    min_action=min_action,
                                                    max_action=max_action,
                                                    device=device,
                                                    args=args).to(device)

    self.env = misc.make_env(args=args, ax=ax, agent_names=agent_names)
    self.args = args

  def run(self):
    total_step = 1
    print(f'test {self.name} {total_step}')
    while self.g_ep.value < self.args.ep_max_timesteps:
      state, _ = self.env.reset()
      buffer_s, buffer_a, buffer_r = {}, {}, {}
      for agent_name in self.local_net_dict.keys():
        buffer_s[agent_name] = []
        buffer_a[agent_name] = []
        buffer_r[agent_name] = []
      ep_r = 0.
      for _ in range(self.args.step_per_ep):
        action_n = {}
        for agent_name, local_net in self.local_net_dict.items():
          # print(f'{state}, {type(state)}')
          action_n[agent_name] = local_net.select_action(state[agent_name])

        new_s, r, done, truncated, _ = self.env.step(action_n)
        ep_r += sum(r.values())
        for agent_name in self.local_net_dict.keys():
          buffer_a[agent_name].append(action_n[agent_name])
          buffer_s[agent_name].append(state[agent_name])
          buffer_r[agent_name].append(r[agent_name])

        if total_step % self.args.a3c_global_update_freq == 0 or done or truncated:
          # update global and assign to local net
          # sync
          push_and_pull(self.optimizer_dict, self.local_net_dict, self.global_net_dict, done,
                        new_s, buffer_s, buffer_a, buffer_r, self.args.discount)
          buffer_s, buffer_a, buffer_r = [], [], []

          if done or truncated:  # done and print information
            record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
            break
        state = new_s
        total_step += 1

    self.res_queue.put(None)
