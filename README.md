# Collaboration_and_Competition
This implementation is based on the OpenAI paper [Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf). In this readme, we will go through the implementation details and hyperparameters selection.

## Environment
![alt text](https://github.com/vjrahil/Collaboration_and_Competition/blob/master/Images/environment.png)<br/>
In this project, 2 agents are required to control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. The goal of each agent is to keep the ball in play without touching the table.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is **at least +0.5**.<br/>
This environment is based on [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) Tennis environment

## Getting Started
### Installing requirement
* You have to clone the project.
* Then you have to install the Unity environment as described in the [Getting Started section](https://github.com/udacity/deep-reinforcement-learning/blob/master/p3_collab-compet/README.md) (The Unity ML-agent environment is already configured by Udacity).
* Download the environment from one of the links below. You need only select the environment that matches your operating system:<br/>
  * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
  * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
  * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
  * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
* Unzip the file in the project environment directory.

### Train the Agent
Execute the Tennis.ipynb file.<br />

## Architecture
The DDPG algorithm uses both the policy network(Actor) and a value based network(Critic). It combines both these network to create a more stable algorithm for tackling contiuous action space problems.
The idea behind using the Critic network is to help the policy network(actor) to reduce its variance. The critic network has low variance and low bias, on the other hand, the actor-network has high variance, so with the help of critic, we try to reduce it. In this project, we train two DDPG agents simultaneously(each representing a racket in the tennis environment). Both these DDPG agents have a similar archutecture.<br />
### Actor Network
* The actor-network takes in states and generates a vector of four, each in the range of (-1,1). For this, we use tanh to keep the final output in this range.
* We use two copies of this network,a local and a target network. This helps in stablising the loss function.<br />

|Layers|Dimensions|Activation Function|
|--------|---------|------------------|
|Linear|(state_size,128)|Relu|
|Linear|(128,256)|Relu|
|Linear|(256,action_size)|Tanh|

### Critic Network
* The critic network takes in states and the actions for each state generated from the Actor-network and uses them to produce a value.
* We use two copies of this network, a local and a target network. This helps in stabilising the loss function.<br />

|Layers|Dimensions|Activation Function|
|------|----------|--------------------|
|Linear|((state_size + action_size)* num_agents,128)|Relu|
|Linear|(128,256)|Relu|
|Linear|(256,1)|None|

### Exploration
In discrete action space, we use a probabilistic action selection policy(epsilon-greedy). But it won’t work in a continuous action space. In the paper of [Continuous control With Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf), the aurthors used [Ornstein Uhlenbeck Process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) to add noise to the action. This process uses the previous noise to prevent the noise from canceling out or “freezing” the overall dynamics.

### Memory
To store the memory, I used a single Replay Buffer of size *int(1e6)*. The idea behind using a single Replay Buffer for two agents is that they will be able to share experiences and gain useful insiht on how to act when the other agent behave ina certain way.
All the experiences were stored as a tuple of the form *(state list,action list ,rewards,next_state list,dones)*.

### Hyperparameter
|Parameters|Values|
---------|--------|
|Actor LR| 1e-4|
|Critic LR| 1e-3|
|TAU| 1e-3|
|BATCH_SIZE| 128|
|BUFFER SIZE| int(1e6)|
|GAMMA| .99|
|Actor optimizer| Adam|
|Critic optimizer|Adam|
|Weight DECAY for Critic optimizer| 0.0|  

Both the agent's weights were updated after **every timestep**.<br />

## Observation

After a lot of hyperprameter finetuning, I was able to solve the environment in 1620 episodes Here are the resultant graphs for the **episodic maximum score** and **the moving verage across the episodes respectively**.<br />
![alt text](https://github.com/vjrahil/Collaboration_and_Competition/blob/master/Images/Episodic_score.png) <br/>
![alt text](https://github.com/vjrahil/Collaboration_and_Competition/blob/master/Images/Moving_average_across_episodes.png)

## Future Experimentation
* Try out different algorithm such as [Multi Agent DQN](https://papers.nips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning.pdf) and [Multi Agent PPO](https://arxiv.org/abs/1707.06347)
* Figuring out a more systematic way of selecting hyperparameters such as Beam Search or Grid Search.


## Repository Instructions
* Clone the directory by using this command.
```
    git clone https://github.com/vjrahil/Collaboration_and_Competition
```
