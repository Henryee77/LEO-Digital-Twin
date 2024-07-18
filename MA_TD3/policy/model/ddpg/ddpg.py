import copy
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from misc.replay_buffer import ReplayBuffer
from ...policy_base import PolicyBase

# Implementation of the Deep Deterministic Policy Gradient algorithm (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim):
    super(Actor, self).__init__()

    self.network = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                 nn.ELU(),
                                 nn.Linear(hidden_dim, hidden_dim // 2),
                                 nn.ELU(),
                                 nn.Linear(hidden_dim // 2, hidden_dim // 4),
                                 nn.ELU(),
                                 nn.Linear(hidden_dim // 4, hidden_dim // 8),
                                 nn.ELU(),
                                 nn.Linear(hidden_dim // 8, hidden_dim // 16),
                                 nn.ELU(),
                                 nn.Linear(hidden_dim // 16, hidden_dim // 32),
                                 nn.ELU(),
                                 nn.Linear(hidden_dim // 32, action_dim))

  def forward(self, state):
    res = self.network(state)
    return torch.tanh(res)


class Critic(nn.Module):
  def __init__(self, critic_input_dim, hidden_dim):
    super(Critic, self).__init__()
    self.q_network = nn.Sequential(nn.Linear(critic_input_dim, hidden_dim),
                                   nn.ELU(),
                                   nn.Linear(hidden_dim, hidden_dim // 2),
                                   nn.ELU(),
                                   nn.Linear(hidden_dim // 2, hidden_dim // 4),
                                   nn.ELU(),
                                   nn.Linear(hidden_dim // 4, hidden_dim // 8),
                                   nn.ELU(),
                                   nn.Linear(hidden_dim // 8, hidden_dim // 16),
                                   nn.ELU(),
                                   nn.Linear(hidden_dim // 16, hidden_dim // 32),
                                   nn.ELU(),
                                   nn.Linear(hidden_dim // 32, 1))

  def forward(self, s, a):
    x = torch.cat([s, a], 1)
    q = self.q_network(x)
    return q


class DDPG(PolicyBase):
  def __init__(self, state_dim, action_dim, device, args):
    self.args = args
    self.device = device
    # Initialize actor networks and optimizer
    self.actor = Actor(state_dim=state_dim,
                       action_dim=action_dim,
                       hidden_dim=args.actor_n_hidden).to(device)
    self.actor_target = copy.deepcopy(self.actor)
    self.actor_optimizer = torch.optim.Adam(
      self.actor.parameters(), lr=args.actor_lr, weight_decay=args.lambda_l2)

    # Initialize critic networks and optimizer
    self.critic = Critic(critic_input_dim=state_dim + action_dim,
                         hidden_dim=args.critic_n_hidden).to(device)
    self.critic_target = copy.deepcopy(self.critic)
    self.critic_optimizer = torch.optim.Adam(
      self.critic.parameters(), lr=args.critic_lr, weight_decay=args.lambda_l2)

    self.batch_size = args.batch_size
    self.discount = args.discount
    self.tau = args.tau

  def select_action(self, state):
    self.actor.eval()

    state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    action = self.actor(state)
    return action.cpu().data.numpy().flatten()

  def train(self,
            replay_buffer: ReplayBuffer,
            total_train_iter: int) -> Dict[str, float]:
    self.actor.train()
    debug = {}

    # Sample from the experience replay buffer
    x, y, u, r, d = replay_buffer.sample(self.batch_size)
    state = torch.FloatTensor(x).to(self.device)
    action = torch.FloatTensor(u).to(self.device)
    next_state = torch.FloatTensor(y).to(self.device)
    not_done = torch.FloatTensor(1 - d).to(self.device)
    reward = torch.FloatTensor(r).to(self.device)

    # Compute the target Q-value
    target_Q = self.critic_target(next_state, self.actor_target(next_state))
    target_Q = reward + (not_done * self.discount * target_Q).detach()

    # Get the current Q-value estimate
    current_Q = self.critic(state, action)

    # Compute the critic loss
    critic_loss = F.mse_loss(current_Q, target_Q)
    debug['critic_loss'] = critic_loss.cpu().data.numpy().flatten() / self.batch_size

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Compute the actor loss
    actor_loss = -self.critic(state, self.actor(state)).mean()
    debug['actor_loss'] = actor_loss.cpu().data.numpy().flatten()

    # Optimize the actor
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # Soft update the target networks
    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    return debug

  # Save the model parameters
  def save(self, filename):
    torch.save(self.critic.state_dict(), filename + '_critic')
    torch.save(self.critic_optimizer.state_dict(), filename + '_critic_optimizer')

    torch.save(self.actor.state_dict(), filename + '_actor')
    torch.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')

  # Load the model parameters
  def load(self, filename):
    self.critic.load_state_dict(torch.load(filename + '_critic'))
    self.critic_optimizer.load_state_dict(torch.load(filename + '_critic_optimizer'))
    self.critic_target = copy.deepcopy(self.critic)

    self.actor.load_state_dict(torch.load(filename + '_actor'))
    self.actor_optimizer.load_state_dict(torch.load(filename + '_actor_optimizer'))
    self.actor_target = copy.deepcopy(self.actor)
