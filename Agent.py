import numpy as np
import torch
from Memory import ReplayBuffer
from OUSample import OUSample
import torch.nn.functional as F

class Agent():
    def __init__(self,config):
        self.config = config
        self.env = config.env
        self.local_memory = None
        #initialize the actor network from config
        self.actor_local = config.actor_local
        self.actor_target = config.actor_target
        self.actor_optim = config.actor_optimizer
        #initialize the critic network
        self.critic_local = config.critic_local
        self.critic_target = config.critic_target
        self.critic_optim = config.critic_optimizer

        self.total_steps = 0
        self.state = None
        if config.noise:
            self.noise = OUSample(self.config.env.action_dim,self.config.seed)

    def get_action(self,np_state):
        #action from local actor network
        state = torch.from_numpy(np_state).float().to(self.config.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy().flatten()
        self.actor_local.train()
        #apply the noise
        actions += self.noise.sample()
        return np.clip(actions,self.config.env.action_space_min , self.config.env.action_space_max)

    def reset(self):
        #reset the noise to mean value
        self.noise.reset()

    def learn(self,states, actions, rewards, next_states, dones):

        # predict next action and Q value from taget Neural Network
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Q targets value with reward of next state
        Q_targets = rewards + (self.config.discount * Q_targets_next * (1 - dones))
        # Compute loss over local and expected values
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # backpropagate the loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # make optimalization step
        self.critic_optim.step()

        # calculate the actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Backpropagate the actor loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        #make optimalization step
        self.actor_optim.step()

        #perform the polyak update of all target networks. (First and second critic network is build in NeuralNetwork class)
        self.polyak_update(self.actor_local,self.actor_target)
        self.polyak_update(self.critic_local, self.critic_target)

    #small update from local to target network
    def polyak_update(self,source,target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.config.polyak_tau * param.data + (1 - self.config.polyak_tau) * target_param.data)

    #save of network
    def save(self, filename):
        torch.save(self.actor_local.state_dict(), '%sactor.pth' % (filename))
        torch.save(self.critic_local.state_dict(), '%scritic.pth' % (filename))