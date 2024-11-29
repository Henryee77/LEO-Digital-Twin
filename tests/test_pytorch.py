from MA_TD3.policy.model import TD3
import argparse
import torch

parser = argparse.ArgumentParser(description='')

# NN Training
parser.add_argument(
    '--model', default='TD3', type=str,
    help='Learnig model')
parser.add_argument(
    '--batch-size', default=16, type=int,
    help='Batch size for both actor and critic')
parser.add_argument(
    '--actor-lr', default=5e-5, type=float,
    help='Learning rate for actor')
parser.add_argument(
    '--critic-lr', default=1e-4, type=float,
    help='Learning rate for critic')
parser.add_argument(
    '--lr-reduce-factor', default=0.9, type=float,
    help='Reduce factor of learning rate')
parser.add_argument(
    '--lr-reduce-patience', default=100, type=int,
    help='Patience of reducing learning rate')
parser.add_argument(
    '--lambda-l2', default=1e-9, type=float,
    help='L2 regularization factor')
parser.add_argument(
    '--clipping-grad-norm', default=1, type=float,
    help='Value of clipping grad norm')
parser.add_argument(
    '--actor-n-hidden', default=3200, type=int,
    help='Number of hidden neuron')
parser.add_argument(
    '--critic-n-hidden', default=6400, type=int,
    help='Number of hidden neuron')
parser.add_argument(
    '--training-period', default=25, type=int,
    help='Peiord (number of timeslot) of NN training.')
parser.add_argument(
    '--replay-buffer-size', default=2000, type=int,
    help='The printing number of the network weight (for debug)')

# --------------- TD3 -----------------------
parser.add_argument(
    '--tau', default=0.01, type=float,
    help='Target network update rate')
parser.add_argument(
    '--policy-freq', default=2, type=int,
    help='Frequency of delayed policy updates')
parser.add_argument(
    '--min-epsilon', default=0.25, type=float,
    help='The minimum of epsilon')
parser.add_argument(
    '--expl-noise', default=0.2, type=float,
    help='The stdv of the exploration noise')
parser.add_argument(
    '--policy-noise', default=0.1, type=float,
    help='The policy noise')
parser.add_argument(
    '--noise-clip', default=0.1, type=float,
    help='The clip range of policy noise')
parser.add_argument(
    '--epsilon-decay-rate', default=0.999, type=float,
    help='The rate of epsilon decay')
parser.add_argument(
    '--discount', default=1e-2, type=float,
    help='Discount factor')
parser.add_argument(
    '--full-explore-steps', default=1e4, type=int,
    help='Number of steps to do exploration')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = TD3(5, 5, 10, [30, 20, 10], [30, 20, 10], 'test',
            args, [-1] * 5, [1] * 5, device)

a_dict = model.actor.state_dict()
print(a_dict)
print('-----------------------------------------------')

for key in a_dict:
  a_dict[key] *= 2

print(model.actor.state_dict())
