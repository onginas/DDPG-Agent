# DRL - DDPG - Reacher Continuous Control
Udacity Deep Reinforcement Learning Nanodegree Program - implementation of Deep Deterministic Policy Gradient algorithm


### Observations:
- There's <b>Report.ipynb</b> file for jupyter notebook execution where is described and showed the implementation of DDPG Agent
- If you are not using a Windows environment, you will follow instruction on this [webpage](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)
- The <b>DDPG_agent.model</b> saved the model with the best average score over all twenty agents
- The necessary python files are below. There's necessary to keep all these files in current workdir
	* network_utils.py
		*utilities for neural network
	
	* network_body.py
		*files with classes for Fully Connected Neural Network or Dummy Body
	
	* network_heads.py
		*file with critic and actor neural network with function for prediction for Q_values or action
	
	* agent_based.py
		* file with base function for each agent
	
	* DDPG_agent.py
		*file with DDPG Agent with functions
	
	* randomProcess.py
		*file with Orstein-Uhlenbeck process for adding noise

### Requeriments:
- tensorflow==1.7.1
- Pillow>=4.2.1
- matplotlib
- numpy>=1.11.0
- jupyter
- pytest>=3.2.2
- docopt
- pyyaml
- protobuf==3.5.2
- grpcio==1.11.0
- pandas
- scipy
- ipykernel
- torch==0.4.0
- seaborn
- matplotlib



## The problem:
- The task phocuses on a continuous control where each of the agent must follow the ball.
- It's a continuous problem where was implemented DDPG Algoritm. There's actor and critic neural network. Actor predicts action (torque of each join) and critic predicts expected Q_values
- The reward of +0.1 is provided for each step that the agent's hand is in the goal location, in this case, the moving ball.
- The goal of this task is to achieve an average score at least 30 over 100 consecutive episodes over 20 agents.


## The solution:
- This taksk is solved by the Deep Deterministic Policy Gradients algorithm.
- This task brought brought me lots of challenges e.g. hyperparameters tunning or creating own DDPG Agent.
- Another thing to highlight is to get inspiration for optimization of other models.
- For the future, it seems that the algorithm works well but I think that the results could be improved by e.g. better architecture of neural network, hyperpameters or using another algorithm (e.g.D4PG).


### The hyperparameters:
- the hyperpameters are in the file <b>DDPG_agent.py</b>.
- The actual configuration of the hyperparameters is: 
  - Learning Rate: 1e-3 (in both DNN)
  - Batch Size: 256
  - Replay Buffer: 1e6
  - Gamma: 0.99
  - Tau: 1e-3
  - Ornstein-Uhlenbeck noise parameters (0.15 theta and 0.2 sigma.)
  - warm-up: 4 (number of episodes before the target network is updated)

- For the neural models:    
  - Actor    
    - Hidden: (input, 64)  - function unit: ReLU
    - Hidden: (128, 64)    - function unit: ReLU
    - Output: (64, 4)      - function unit: TanH

  - Critic
    - Hidden: (input + action_size, 64)	- function unit: ReLU
    - Hidden: (128, 64)  				- function unit: ReLU
    - Output: (64, 1)                  	- function unit: Linear
