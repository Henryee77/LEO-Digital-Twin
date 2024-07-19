"""async_on_policy_trainer.py"""
import os
from typing import Dict
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch .multiprocessing as mp

from agent import Agent
from policy.model.a3c import ActorCritic
from policy.model.a3c import Worker
from policy.model.a3c import SharedAdam

os.environ["OMP_NUM_THREADS"] = "1"


class AsyncTrainer(object):

  def __init__(self, args):
    self.args = args

  def train(self, agent_dict: Dict[str, Agent], ax, log, tb_writer):
    temp_agent = agent_dict[next(iter(agent_dict))]
    state_dim = temp_agent.state_dim
    action_dim = temp_agent.action_dim
    manager = mp.Manager()
    global_net_dict = manager.dict()
    optimizer_dict = manager.dict()
    for agent_name, agent in agent_dict.items():
      agent.policy.share_memory()
      optimizer_dict[agent_name] = SharedAdam(agent.policy.parameters(), lr=(self.args.actor_lr + self.args.critic_lr) / 2,
                                              betas=(0.95, 0.999))  # global optimizer
      global_net_dict[agent_name] = agent.policy
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    worker_num = 1  # mp.cpu_count()
    log[self.args.log_name].info(f'The number of workers: {worker_num}')
    workers = [Worker(state_dim=state_dim, action_dim=action_dim, actor_hidden_dim=self.args.actor_n_hidden, critic_hidden_dim=self.args.critic_n_hidden,
                      min_action=temp_agent.min_action, max_action=temp_agent.max_action, global_net_dict=global_net_dict, optimizer_dict=optimizer_dict,
                      global_ep=global_ep, global_ep_r=global_ep_r, res_queue=res_queue, ax=ax, device=temp_agent.device, args=self.args, name=i)
               for i in range(worker_num)]
    [w.start() for w in workers]

    total_eps = 1

    while True:
      r = res_queue.get()
      if r is not None:
        tb_writer.add_scalars(
          f'reward', {'reward': r}, total_eps)
        total_eps += 1
        log[self.args.log_name].info(f'The appended reward: {r}')
      else:
        break
      del r
    [w.join() for w in workers]
