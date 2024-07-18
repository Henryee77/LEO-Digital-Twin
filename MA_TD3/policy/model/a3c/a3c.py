"""a3c.py"""

import math
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
  def __init__(self, state_dim, action_dim, actor_hidden_dim, critic_hidden_dim, min_action, max_action, device, args):
    super(ActorCritic, self).__init__()
    self.main_actor = nn.Sequential(nn.Linear(state_dim, actor_hidden_dim),
                                    nn.ELU(),
                                    nn.Linear(actor_hidden_dim, actor_hidden_dim // 2),
                                    nn.ELU(),
                                    nn.Linear(actor_hidden_dim // 2, actor_hidden_dim // 4),
                                    nn.ELU())
    self.mu = nn.Sequential(nn.Linear(actor_hidden_dim // 4, actor_hidden_dim // 8),
                            nn.ELU(),
                            nn.Linear(actor_hidden_dim // 8, actor_hidden_dim // 16),
                            nn.ELU(),
                            nn.Linear(actor_hidden_dim // 16, actor_hidden_dim // 32),
                            nn.ELU(),
                            nn.Linear(actor_hidden_dim // 32, action_dim))
    self.sigma = nn.Sequential(nn.Linear(actor_hidden_dim // 4, actor_hidden_dim // 8),
                               nn.ELU(),
                               nn.Linear(actor_hidden_dim // 8, actor_hidden_dim // 16),
                               nn.ELU(),
                               nn.Linear(actor_hidden_dim // 16, actor_hidden_dim // 32),
                               nn.ELU(),
                               nn.Linear(actor_hidden_dim // 32, action_dim))

    self.critic = nn.Sequential(nn.Linear(state_dim, critic_hidden_dim),
                                nn.ELU(),
                                nn.Linear(critic_hidden_dim, critic_hidden_dim // 2),
                                nn.ELU(),
                                nn.Linear(critic_hidden_dim // 2, critic_hidden_dim // 4),
                                nn.ELU(),
                                nn.Linear(critic_hidden_dim // 4, critic_hidden_dim // 8),
                                nn.ELU(),
                                nn.Linear(critic_hidden_dim // 8, critic_hidden_dim // 16),
                                nn.ELU(),
                                nn.Linear(critic_hidden_dim // 16, critic_hidden_dim // 32),
                                nn.ELU(),
                                nn.Linear(critic_hidden_dim // 32, 1))

    self.distribution = torch.distributions.Normal
    self.action_low_b = min_action
    self.action_upper_b = max_action
    self.device = device
    self.args = args

  def forward(self, x):
    return torch.tanh(self.mu(self.main_actor(x))), F.softplus(self.sigma(self.main_actor(x))) + 1e-6, self.critic(x)

  def select_action(self, state):
    state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    mu, sigma = (torch.tanh(self.mu(self.main_actor(state))),
                 F.softplus(self.sigma(self.main_actor(state))) + 1e-6)
    action = self.distribution(mu.data, sigma.data)
    return np.clip(action.sample().cpu().data.numpy().flatten(), self.action_low_b, self.action_upper_b)

  def loss_func(self, s: torch.Tensor, a: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
    self.train()
    mu, sigma, values = self.forward(s)
    advantage = v_t - values
    value_loss = torch.pow(advantage, 2)

    m = self.distribution(mu, sigma)
    log_prob = m.log_prob(a)
    policy_loss = -(advantage * log_prob)
    entropy_loss = m.entropy()
    return self.vf_coef * value_loss + policy_loss + self.args.ent_coef * entropy_loss

  def save(self, filename, directory):
    torch.save(self.state_dict(), f'{directory}/{filename}_critic.pth')

  def load(self, filename, directory):
    self.load_state_dict(torch.load(f'{directory}/{filename}_actor.pth'))
