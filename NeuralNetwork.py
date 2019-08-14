import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def hidden_init(layer):
    #change initialization of weight from original https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
    #from input size to output size
    #this reduce the value of init weights
    #and make NN mutch smoother
    lim = 1. / math.sqrt(layer.weight.size(0))
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size,action_size,seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes for layers
            seed (int): Random seed
            gate (function): activation function
            final_gate (function): final activation function
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.normalizer = nn.BatchNorm1d(state_size)
        self.input = nn.Linear(state_size, 500)
        self.hidden1 = nn.Linear(500,300)
        self.output = nn.Linear(300, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.input.weight.data.uniform_(*hidden_init(self.input))
        self.hidden1.weight.data.uniform_(*hidden_init(self.hidden1))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.normalizer(states)
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = torch.tanh(self.output(x))
        return x

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size,action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes for layers
            seed (int): Random seed
            gate (function): activation function
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dropout = nn.Dropout(p=0.2)
        self.normalizer = nn.BatchNorm1d(state_size)
        self.input = nn.Linear(state_size, 500)
        self.hidden1_action = nn.Linear(500+action_size, 300)
        self.hidden2 = nn.Linear(300, 200)
        self.output = nn.Linear(200,1)
        self.reset_parameters()

    def reset_parameters(self):
        self.input.weight.data.uniform_(*hidden_init(self.input))
        self.hidden1_action.weight.data.uniform_(*hidden_init(self.hidden1_action))
        self.hidden2.weight.data.uniform_(*hidden_init(self.hidden2))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.normalizer(states)
        x = F.relu(self.input(x))
        y = torch.cat((x, actions), dim=1)
        y = F.relu(self.hidden1_action(y))
        y = F.relu(self.hidden2(y))
        y = self.dropout(y)
        return self.output(y)
