# Multi Agent - Tennis
### Introduction
This project, describes the reinforcement learning to resolve the continues control problem with multi agents.
The problem is describe with continues state space and continues action space.
The goal is to play tennis with two agents :) 
I used the DDPG algorithm with initial weights set up. 

The enviroment comes from Unity, please read the Unity Environment, before making a copy and trying yourself!

### Get Started 
Clone the repository, install the Unity Enviroment and start with ExperienceManager.py (update UNITY_ENVIROMENT before run)

### Enviroment description
**You can see trained Wimbledon in action [here](https://www.youtube.com/watch?v=lie4xXOuJL4)**

This environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Project structure
The project in writen in python and using the pytorch framework for Deep Neural network. More requirements read bellow.
The project files are following:
- ExperienceManager.py - the main file.  Responsible for run the experience, run episode, interact with MultiAgent and store statistic of episode reward 
- MultiAgent.py - is the wrapper over agents. It calls the agents for action and call learn over each agent
- Agent.py - responsible for choosing action in particular state, interact with memory, and learning process             
- Memory.py - class is reponsible for storing data in array data structure, and randomly sampling data from it
- NeuralNetwork.py - define the Neural network model in Pytorch, For both Actor and Critic NN. 
- EnvironmentWrapper.py -  responsible for creating and interaction with Unity env. 
- Config.py - class hold the all hyperparams and object required by agent
- Util.py - miscellaneous functions like, prepare model file name, store graph
- actor.pth - the learned neural network model, for interacting actions
- critic.pth - the learned neural network model provides the Q-Value function
- Report.pdf - describe the work process and some hyper parameter testing

### Installation requirement
The project was tested on 3.6 python and requires the following packages to be installed:
- numpy 1.16.4
- torch 1.1.0
- matplotlib 3.1.0
- unityagent 0.4.0

### Unity Environment
After success instalation please navigate to line num 13 in ExperienceManager.py and update the path to your installation directory
To try it yourself and see how wise you agent can be :), you'll need to download a new Unity environment.
You need only select the environment that matches your operating system:

* Linux: [download here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [download here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [download here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [download here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
