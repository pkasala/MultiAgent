from Agent import Agent
import numpy as np


class MultiAgent:
    def __init__(self, config):
        self.config = config
        self.agents =[]
        for _  in range(config.num_agents):
            #create the agents
            self.agents.append(Agent(self.config))

        self.total_steps = 0
        self.states = None

    def get_action(self, all_states):
        actions=[]
        #perform action for each agent
        for agent, states in zip(self.agents, all_states):
            #convert vector to array
            st = np.expand_dims(states, axis=0)
            #get action for agent
            actions.append(agent.get_action(np.expand_dims(states, axis=0)))
        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory, separate agents experiences into each-one observation
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.config.memory.add(state, action, reward, next_state, done)

        self.total_steps+=1
        # Learn every UPDATE_EVERY time steps.
        if self.total_steps > self.config.warm_up and self.total_steps % self.config.learn_every == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.config.memory.size() > self.config.memory_batch_size:
                for agent in self.agents:
                    states, actions, rewards, next_states, dones  = self.config.memory.sample()
                    agent.learn(states, actions, rewards, next_states, dones)

    #save the save of network
    def save(self, filename):
        for index,agent in enumerate(self.agents):
            agent.save(filename=filename+"_"+str(index))