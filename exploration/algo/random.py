import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RandomAgent:
    def __init__(self,
                 ):
        pass

    def update(self, rollouts):
        return 0, 0, 0


class RandomPolicy(object):
    def __init__(self, env, num_processes):
        super(RandomPolicy, self).__init__()
        self.env = env
        self.num_processes = num_processes

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return 1

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        action = torch.tensor([self.env.action_space.sample() for _ in range(self.num_processes)]).reshape(self.num_processes, -1)
        a = torch.zeros(self.num_processes, 1).float()
        return a, action, a, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        c = torch.Tensor([0.]).float().unsqueeze(0)
        
        return c

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        return 0, 0, 0, 0