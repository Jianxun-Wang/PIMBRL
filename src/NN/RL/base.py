import numpy as np
import gym
import torch
import torch.nn as nn
from utility.utils import check_dict_valid

def mlp(qp:str,
        activation, 
        obs_dim:int,
        act_dim:int,
        hidden_sizes:list, 
        output_activation,):

    sizes_dict = {'p':[obs_dim] + hidden_sizes + [act_dim],
                  'q':[obs_dim + act_dim] + hidden_sizes + [1],
                  'c':[obs_dim] + hidden_sizes}
    check_dict_valid(qp,sizes_dict)
    if qp == 'q': output_activation = nn.Identity
    sizes = sizes_dict[qp]
    
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def cnn1d():
    raise NotImplementedError



######################### Actor Network #########################
class MLPActor(nn.Module):
    def __init__(self, act_limit, **para):
        super().__init__()
        self.act_min = act_limit[:,0]
        self.act_len = act_limit[:,1]-act_limit[:,0]
        self.net = mlp(qp='p',**para)

    def forward(self, obs):
        return self.act_min+self.act_len*(self.net(obs)+1)/2 

######################### Q Function ############################
class MLPQ(nn.Module):
    def __init__(self, **para):
        super().__init__()
        self.net = mlp(qp='q',**para)

    def forward(self, obs, act):
        q = self.net(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


###################### Actor-Critic ##############################
class ActorCritic(nn.Module):
    '''
        Base class for actor-critic structure 
    '''
    def __init__(self, act_limit, act_space_type:str):
        super().__init__()
        act_space_types = {'d':self.__get_action_d__,'c':self.__get_action_c__}
        check_dict_valid(act_space_type,act_space_types)
        self.get_action = act_space_types[act_space_type]
        self.act_limit = act_limit
        self.act_min = act_limit[:,0]
        self.act_len = act_limit[:,1]-act_limit[:,0]
        
    
    

if __name__=="__main__":
    pass
    