from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


from .base import MLPActor, ActorCritic, MLPQ, mlp
from utility.utils import check_dict_valid, clip_tensor



TENSOR2 = torch.tensor(2)
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class sActorCritic(ActorCritic):
    '''
        Deterministic variant of ActorCritic
    '''
    def act(self, obs,stochastic):
        with torch.no_grad():
            return self.pi(obs,stochastic)

    def __get_action_d__(self,o, stochastic=True):
        a,_ = self.act(o,stochastic)
        return clip_tensor(a,self.act_limit[:,0],self.act_limit[:,1]).round().int()
    
    def __get_action_c__(self,o, stochastic=True):
        a,_ = self.act(o, stochastic)
        return clip_tensor(a,self.act_limit[:,0],self.act_limit[:,1])


class GaussianMLPActor(MLPActor):
    def __init__(self, act_limit, **para):
        super().__init__(act_limit, **para)
        self.net = mlp(qp='c', **para)
        hidden_sizes, act_dim = para['hidden_sizes'], para['act_dim']
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        # self.act_limit = act_limit

    def forward(self, obs, stochastic=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        if (log_std>LOG_STD_MAX).any():
            print('std clamped')
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if stochastic:
            pi_action = pi_distribution.rsample()

            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)\
             - (2*(torch.log(TENSOR2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
        
            pi_action = torch.tanh(pi_action)
            pi_action = self.act_len * pi_action

            return pi_action, logp_pi

        else:
            # Only used at test.
            pi_action = torch.tanh(mu)
            pi_action = self.act_len * pi_action
            return pi_action, None
            
        
        


class SAC_net(sActorCritic):

    def __init__(self, act_limit, act_space_type:str, 
                 Actor_type, Actor_para:dict,
                 Q_type, Q_para:dict):
        
        super().__init__(act_limit, act_space_type)
        self.pi = Actor_type(act_limit,**Actor_para)
        self.q1 = Q_type(**Q_para)
        self.q2 = Q_type(**Q_para)

             