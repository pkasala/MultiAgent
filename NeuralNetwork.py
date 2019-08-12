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

    def __init__(self, state_size,action_size,seed):
        super(Actor, self).__init__()
        #use the global seed
        self.seed = torch.manual_seed(seed)
        #include the batch normalization for becasue we normalize the batch input data, and therfore reduce internal covariant
        self.normalizer = nn.BatchNorm1d(state_size)
        #basic full connected input layer
        self.input = nn.Linear(state_size, 512)
        self.hidden1 = nn.Linear(512,256)
        self.output = nn.Linear(256, action_size)
        #this ensure to init the weight closer to 0
        self.reset_parameters()

    def reset_parameters(self):
        self.input.weight.data.uniform_(*hidden_init(self.input))
        self.hidden1.weight.data.uniform_(*hidden_init(self.hidden1))
        #fixed values of output layer - this is too small for standard init
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        #normalize the batch input
        x = self.normalizer(states)
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        #output action is in -1 +1 - use tanh
        x = torch.tanh(self.output(x))
        return x

class Critic(nn.Module):

    def __init__(self, state_size,action_size, seed):
        super(Critic, self).__init__()
        #use global seed
        self.seed = torch.manual_seed(seed)
        self.dropout = nn.Dropout(p=0.2)
        # include the batch normalization for becasue we normalize the batch input data, and therfore reduce internal covariant
        self.normalizer = nn.BatchNorm1d(state_size)
        self.input = nn.Linear(state_size, 512)
        #adjust the network with action state
        self.hidden1_action = nn.Linear(512+action_size, 256)
        self.output = nn.Linear(256,1)
        # this ensure to init the weight closer to 0
        self.reset_parameters()

    def reset_parameters(self):
        self.input.weight.data.uniform_(*hidden_init(self.input))
        self.hidden1_action.weight.data.uniform_(*hidden_init(self.hidden1_action))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        #normalize the input batch
        x = self.normalizer(states)
        #feed forward layers
        x = F.relu(self.input(x))
        y = torch.cat((x, actions), dim=1)
        y = F.relu(self.hidden1_action(y))
        y = self.dropout(y)
        return self.output(y)
