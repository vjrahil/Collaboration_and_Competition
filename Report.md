# Collaboration_and_Competition
## Learning algorithm
Implemented Multi Agent Deep Deterministic Policy Gradient(MADDPG) algorithm to solve the Unity Tennis environment. Go through the **Method section** of the paper for understanding the algorithm details. Here is the image of the information flow(interaction between actor and critic) in the algorithm. <br />
![alt text](https://github.com/vjrahil/Collaboration_and_Competition/blob/master/Images/maddpg.png)
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

