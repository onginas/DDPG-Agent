# -*- coding: utf-8 -*-
"""
Function for Deterministic Critic Network
"""
from utils.network_utils import *
from utils.network_body import *



class DeterministicActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim,
                 actor_optim_fn, critic_optim_fn,
                 phi_body=None, actor_body=None, critic_body=None, device = 'cpu'):
        
        # initializes the parent class object into the child class
        super(DeterministicActorCriticNet, self).__init__()
        
        # create dummy bodies in case that some body is not defined 
        if phi_body is None:
            phi_body = DummyBody(state_dim)
        
        if actor_body is None:
            actor_body = DummyBody(state_dim)
        
        if critic_body is None:
            critic_body = DummyBody(state_dim)
        
        #initialize input bodies    
        self.phi_body = phi_body.to(device)
        self.actor_body = actor_body.to(device)
        self.critic_body = critic_body.to(device)
        self.device = device
            
        # init output layers
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim).to(device), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1).to(device), 1e-3)
            
        # concatenate params for actor and critic
        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        # set optimization for actor and critic
        self.actor_opt = actor_optim_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_optim_fn(self.critic_params + self.phi_params)
        
    # select feature from data
    def feature(self, obs):
        obs = torch.tensor(obs).to(self.device)
        return(self.phi_body(obs).to(self.device))
    
    # prediction for actor's neural network
    def action(self, phi):
        return torch.tanh(self.fc_action(self.actor_body.forward(phi))).to(self.device)
    
    # prediction for critic's neural network
    def critic(self, phi, a):
        tensor = torch.cat((phi.float(), a.float()), dim=1).to(self.device)
        return self.fc_critic(self.critic_body.forward(tensor)).to(self.device)
    