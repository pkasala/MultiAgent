import copy
import numpy as np
import random


class OUSample:
    #Ornstein-Uhlenbeck

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        #mean values of action of size of action space
        #the mean value of action is of course 0
        self.mu = mu * np.ones(size)
        #describe the variance of shock in Ornstein-Uhlenbeck process
        self.theta = theta
        #discribe the variance of modeling process.
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        #reset the state back to mean values
        self.state = copy.copy(self.mu)

    def sample(self):
        #update the state and simulate the Brownian move
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state