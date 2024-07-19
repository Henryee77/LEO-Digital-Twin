"""policy_base.py"""


class PolicyBase(object):
  def __init__(self):
    pass

  def train(self, replay_buffer, total_train_iter):
    """Train the network from replay buffer

    Args:
        replay_buffer (ReplayBuffer): The replay buffer
        total_train_iter (int): The total number of training iteration

    Returns:
        Dict[str, float]: The dict of log
    """
    pass
