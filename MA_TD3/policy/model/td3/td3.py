"""Modified Twin Delayed Deep Deterministic Policy Gradients (TD3)
TD3 Ref: https://github.com/sfujim/TD3
"""
import copy
from typing import Dict, Tuple, List
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....misc.replay_buffer import ReplayBuffer
from ....misc import misc
from ...policy_base import PolicyBase


class Actor(nn.Module):
  """The Actor class"""

  def __init__(self, actor_input_dim, actor_output_dim, hidden_nodes, name):
    super(Actor, self).__init__()
    activ_func = nn.ELU()
    network_dict = misc.construct_dnn_dict(input_dim=actor_input_dim,
                                           output_dim=actor_output_dim,
                                           hidden_nodes=hidden_nodes,
                                           activ_func=activ_func)

    """network_dict = OrderedDict([('fc_1', nn.Linear(actor_input_dim, n_hidden)),
                                ('elu_1', nn.ELU()),
                                ('fc_2', nn.Linear(n_hidden, n_hidden // 2)),
                                ('elu_2', nn.ELU()),
                                ('fc_3', nn.Linear(n_hidden // 2, n_hidden // 4)),
                                ('elu_3', nn.ELU()),
                                ('fc_4', nn.Linear(n_hidden // 4, n_hidden // 8)),
                                ('elu_4', nn.ELU()),
                                ('fc_5', nn.Linear(n_hidden // 8, n_hidden // 16)),
                                ('elu_5', nn.ELU()),
                                ('fc_6', nn.Linear(n_hidden // 16, n_hidden // 32)),
                                ('elu_6', nn.ELU()),
                                ('fc_7', nn.Linear(n_hidden // 32, actor_output_dim))
                                ]) """

    self.layer_num = (len(network_dict) + 1) / 2
    self.network = nn.Sequential(network_dict)
    self.name = name

  def forward(self, state):
    res = self.network(state)
    # print(res)
    return torch.tanh(res)


class Critic(nn.Module):
  """The Critic class"""

  def __init__(self, critic_input_dim, hidden_nodes, name):
    super(Critic, self).__init__()
    activ_func = nn.ELU()

    q_network_dict = misc.construct_dnn_dict(input_dim=critic_input_dim,
                                             output_dim=1,
                                             hidden_nodes=hidden_nodes,
                                             activ_func=activ_func)

    self.q_network_num = 2
    self.layer_num = (len(q_network_dict) + 1) / 2

    self.q1_network = nn.Sequential(q_network_dict)
    self.q2_network = nn.Sequential(copy.deepcopy(q_network_dict))

    self.name = name

  def forward(self, x, u):
    xu = torch.cat([x, u], 1)

    x1 = self.q1_network(xu)
    x2 = self.q2_network(xu)

    return x1, x2

  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)

    x1 = self.q1_network(xu)

    return x1


class TD3(PolicyBase):
  """The TD3 class"""

  def __init__(self, actor_input_dim, actor_output_dim, critic_input_dim, actor_hidden_nodes, critic_hidden_nodes, name, args, action_low, action_high, device):
    super(TD3, self).__init__()
    self.actor = Actor(
        actor_input_dim=actor_input_dim,
        actor_output_dim=actor_output_dim,
        hidden_nodes=actor_hidden_nodes,
        name=name + '_actor').to(device)
    self.actor_target = copy.deepcopy(self.actor)

    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, weight_decay=args.lambda_l2)
    self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      self.actor_optimizer, factor=args.lr_reduce_factor, patience=args.lr_reduce_patience, eps=1e-10)

    self.critic = Critic(critic_input_dim=critic_input_dim,
                         hidden_nodes=critic_hidden_nodes,
                         name=name + '_critic').to(device)
    self.critic_target = copy.deepcopy(self.critic)

    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, weight_decay=args.lambda_l2)
    self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      self.critic_optimizer, factor=args.lr_reduce_factor, patience=args.lr_reduce_patience, eps=1e-10)

    self.name = name
    self.args = args
    self.device = device
    self.action_low = torch.tensor(action_low).to(device)
    self.action_high = torch.tensor(action_high).to(device)
    self.batch_size = args.batch_size
    self.discount = args.discount
    self.tau = args.tau
    self.policy_freq = args.policy_freq

    a_layer_num = iter([p.numel() for p in self.actor.parameters() if p.requires_grad])
    c_layer_num = iter([p.numel() for p in self.critic.parameters() if p.requires_grad])
    self.__actor_layer_param_num = [x + y for x, y in zip(a_layer_num, a_layer_num)]
    self.__critic_layer_param_num = [x + y for x, y in zip(c_layer_num, c_layer_num)]
    self.__nn_param_num = (sum(self.actor_layer_param_num) +
                           sum(self.critic_layer_param_num))

  @ property
  def actor_layer_num(self):
    return self.actor.layer_num

  @ property
  def critic_layer_num(self):
    return self.critic.layer_num

  @ property
  def q_network_num(self):
    return self.critic.q_network_num

  @property
  def actor_layer_param_num(self) -> List[int]:
    return self.__actor_layer_param_num

  @property
  def critic_layer_param_num(self) -> List[int]:
    return self.__critic_layer_param_num

  @property
  def nn_param_num(self) -> int:
    return self.__nn_param_num

  def select_action(self, state):
    state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

    action = self.actor(state)
    return action.cpu().detach().numpy().flatten()

  def train(self,
            replay_buffer: ReplayBuffer,
            total_train_iter: int) -> Dict[str, float]:

    debug = {}

    # Sample replay buffer
    x, y, u, r, d = replay_buffer.sample(self.batch_size)
    state = torch.FloatTensor(x).to(self.device)
    action = torch.FloatTensor(u).to(self.device)
    next_state = torch.FloatTensor(y).to(self.device)
    not_done = torch.FloatTensor(1 - d).to(self.device)
    reward = torch.FloatTensor(r).to(self.device)
    # print(f'data: {state}, {action}, {next_state}, {not_done}, {reward}')

    with torch.no_grad():
      # Select next action according to policy
      noise = torch.clamp(torch.randn_like(action) * self.args.policy_noise,
                          -self.args.noise_clip,
                          self.args.noise_clip)
      next_action = torch.clamp(self.actor_target(next_state) + noise,
                                self.action_low,
                                self.action_high)
      next_action.to(torch.float32)
      # next_action = self.actor_target(next_state)
      # print(f'na: {next_action.size()}')
      # next_action = onehot_from_logits(next_action)

      # Compute the target Q value
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      target_Q = torch.min(target_Q1, target_Q2)
      target_Q = reward + (not_done * self.discount * target_Q).detach()

    # Get current Q estimates
    current_Q1, current_Q2 = self.critic(state, action)
    # print(f'Q: {current_Q1}, {current_Q2}')

    # Compute critic loss
    critic_loss = F.mse_loss(current_Q1, target_Q) + \
        F.mse_loss(current_Q2, target_Q)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.clipping_grad_norm)
    self.critic_optimizer.step()
    self.critic_scheduler.step(critic_loss)
    debug['critic_loss'] = critic_loss.cpu().detach().numpy().flatten() / self.batch_size

    # Delayed policy updates
    if total_train_iter % self.policy_freq == 0:
      # Compute actor loss
      actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

      # Optimize the actor
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      # print(f'weight: {self.actor.network[0].weight}')
      # print(f'grad: {self.actor.network[0].weight.grad}')
      # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.clipping_grad_norm)
      self.actor_optimizer.step()
      self.actor_scheduler.step(actor_loss)
      debug['actor_loss'] = actor_loss.cpu().detach().numpy().flatten()

      # Update the frozen target models
      for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        target_param.data.copy_(
            self.tau * param.data + (1 - self.tau) * target_param.data)

      for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        target_param.data.copy_(
            self.tau * param.data + (1 - self.tau) * target_param.data)

    if total_train_iter % 100 == 0:
      critic_lr, actor_lr = self.get_lr()
      debug['critic_lr'] = critic_lr
      debug['actor_lr'] = actor_lr
    return debug

  def get_lr(self) -> Tuple[float, float]:
    critic_lr, actor_lr = 0, 0
    critic_param_num, actor_param_num = 0, 0

    for param_group in self.critic_optimizer.param_groups:
      critic_lr += param_group['lr']
      critic_param_num += 1
    for param_group in self.actor_optimizer.param_groups:
      actor_lr += param_group['lr']
      actor_param_num += 1

    return (critic_lr / critic_param_num), (actor_lr / actor_param_num)

  def load_actor_state_dict(self, state_dict):
    self.actor.load_state_dict(state_dict=state_dict)

  def load_critic_state_dict(self, state_dict):
    self.critic.load_state_dict(state_dict=state_dict)

  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), f'{directory}/{filename}_actor.pth')
    torch.save(self.critic.state_dict(), f'{directory}/{filename}_critic.pth')

  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load(f'{directory}/{filename}_actor.pth'))
    self.actor_target = copy.deepcopy(self.actor)

    self.critic.load_state_dict(torch.load(f'{directory}/{filename}_critic.pth'))
    self.critic_target = copy.deepcopy(self.critic)
