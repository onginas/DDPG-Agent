# -*- coding: utf-8 -*-
"""
Utils for creating bodies neral netowrks
 
"""
from network_utils import *

class FCBody(nn.Module):
    
    def __init__(self, state_dim, hidden_units = (128, 128, 64), function_unit = F.relu):
        
        # initializes the parent class object into the child class
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        
        # create module list with layers
        self.layers = nn.ModuleList(
             [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        
        # define function unit
        self.function_unit = function_unit
        self.feature_dim = dims[-1]
    
    # forward function for prediction
    def forward(self, x):
        for layer in self.layers:
            x = self.function_unit(layer(x))
        return x
        
class DummyBody(nn.Module):
    
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim
        
    def forward(self, x):
        return x
        