# -*- coding: utf-8 -*-
"""
Script for DDPG Agent and Replay buffer
"""

from collections import deque, namedtuple
import random
import numpy as np
from utils.randomProcess import *
from utils.agent_based import *
from utils.network_body import *
from utils.network_heads import *
from utils.network_utils import *
import torch.nn.functional as F
import torch

"""
DDPG Agent algorithm
"""

class DDPGAgent(BaseAgent):
    
    # initialize agent object
    def __init__(self, state_dim, action_dim, device='cpu'):
        
        # initialize parent object class into children object
        super(BaseAgent, self).__init__()
        
        # set basic parameters of agent
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = 0.99
        self.target_network_mix = 1e-3
        self.device = device
        self.n_step = 0
        self.warm_up = 4
        self.buffer_size = int(1e6)
        self.batch_size = 256 
        self.seed = 0
        
        # create local network
        self.network =  DeterministicActorCriticNet(self.state_dim, self.action_dim,
                                                                   actor_optim_fn  = lambda params: torch.optim.Adam(params, lr=1e-3),
                                                                   critic_optim_fn = lambda params: torch.optim.Adam(params, lr=1e-3), 
                                                                   actor_body = FCBody(self.state_dim, hidden_units = (64, 128, 64), function_unit = F.relu),
                                                                   critic_body = FCBody(self.state_dim + self.action_dim, hidden_units = (64, 128, 64), function_unit = F.relu), device=device)
        # create target network
        self.target_network = DeterministicActorCriticNet(self.state_dim, self.action_dim,
                                                                   actor_optim_fn  = lambda params: torch.optim.Adam(params, lr=1e-3),
                                                                   critic_optim_fn = lambda params: torch.optim.Adam(params, lr=1e-3), 
                                                                   actor_body = FCBody(self.state_dim, hidden_units = (64, 128, 64), function_unit = F.relu),
                                                                   critic_body = FCBody(self.state_dim + self.action_dim, hidden_units = (64, 128, 64), function_unit = F.relu), device=device)
        
        # copy parameters to target network from local network
        self.target_network.load_state_dict(self.network.state_dict())
        
        # add noise from Ornstein-Uhlenbeck process
        self.random_process_fn = OrnsteinUhlenbeckProcess(
                                            size=(self.action_dim, ),
                                            std=0.2)
        
        # create replay buffer
        self.replay_buffer = ReplayBuffer(action_size=self.action_dim, buffer_size=self.buffer_size, batch_size=self.batch_size, seed=self.seed, device = self.device)
        
    # function for copy all parameters between models    
    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1 - self.target_network_mix) + param*self.target_network_mix).to(self.device)
    
    # function for selection of action
    def select_action(self, state, noise = True):
        # convert input from numpy to tensor
        state = torch.from_numpy(state).float().to(self.device)
        
        # predict action from local neural network
        self.network.eval()
        with torch.no_grad():
            action = self.network.action(state).to(self.device)
        self.network.train()
        
        # convert action into numpy
        action = to_np(action)
        
        # add noise to
        if noise:
            action = action + self.random_process_fn.sample()
        action = np.clip(action, -1, 1)
        return action
    
    # make step in learning process
    def step(self, state, action, reward, next_state, done):
        # increment number step
        self.n_step += 1
        
        # add the experience into buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
       
        # run learning process if the length of buffer is same or higher than batch size
        if self.replay_buffer.__len__() >= self.replay_buffer.batch_size:
            self.learn()
    
    # function for learning process of agent
    def learn(self):
        
        # choose random sample from object
        b_states, b_actions, b_rewards, b_next_states, b_dones = self.replay_buffer.sample()
        
        # predict next action from next states
        with torch.no_grad():
            b_next_actions = self.target_network.action(b_next_states)
                    
        # Compute Q_targets for current states (y_i)
        Q_targets =  b_rewards + self.discount * self.target_network.critic(b_next_states, b_next_actions)*(1 - b_dones)
        
        # calculate Q_expected from local network
        Q_expected = self.network.critic(b_states, b_actions)
        
        # calculate losses
        critic_losses = F.mse_loss(Q_targets, Q_expected).to(self.device)
        
        # minimize the critic losses
        self.network.zero_grad()
        critic_losses.backward()
        torch.nn.utils.clip_grad_norm_(self.network.critic_params, 1)
        self.network.critic_opt.step()
            
        # Compute actor loss
        actions_pred = self.network.action(b_states)            
        
        # maximizethe losses
        action_losses = -(self.network.critic(b_states, actions_pred)).mean().to(self.device)
        self.network.zero_grad()
        action_losses.backward()
        self.network.actor_opt.step()
        
        # update params of target neural network
        if self.n_step % self.warm_up  == 0:
            self.soft_update(self.target_network, self.network)
 
    
     
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    # Initialize a ReplayBuffer object.
    def __init__(self, action_size, buffer_size, batch_size, seed, device):

        # parameters initialization
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    # function for adding experience into buffer
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    # function to generate random sample
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # convert outputs into tensor
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    # function to get length of the memory
    def __len__(self):
        return len(self.memory)
        