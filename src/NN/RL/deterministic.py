import torch
import torch.nn as nn

from utility.utils import clip_tensor
from .base import MLPActor,ActorCritic,MLPQ


class dActorCritic(ActorCritic):
    '''
        Deterministic variant of ActorCritic
    '''
    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs)

    def __get_action_d__(self,o, noise_scale=0):
        a = self.act(o)
        a += noise_scale*torch.randn(a.shape)
        return clip_tensor(a,self.act_limit[:,0],self.act_limit[:,1]).round().int()
    
    def __get_action_c__(self,o, noise_scale=0):
        a = self.act(o)
        a += noise_scale*torch.randn(a.shape)
        return clip_tensor(a,self.act_limit[:,0],self.act_limit[:,1])


class DDPG_net(dActorCritic):
    '''
        networks for DDPG
    '''
    def __init__(self, 
                 act_limit, act_space_type:str, 
                 Actor_type:MLPActor, Actor_para:dict,
                 Q_type=MLPQ, Q_para:dict=None):
        # check_dict_valid(net_type,NET_TYPE)
        super().__init__(act_limit, act_space_type)
        # build policy and value function
        self.pi = Actor_type(act_limit,**Actor_para)
        self.q = Q_type(**Q_para)


class TD3_net(dActorCritic):
    '''
        networks for TD3
    '''
    def __init__(self, act_limit, act_space_type:str, 
                 Actor_type:MLPActor, Actor_para:dict,
                 Q_type=MLPQ, Q_para:dict=None):
        # check_dict_valid(net_type, NET_TYPE)
        super().__init__(act_limit, act_space_type)
        # build policy and value functions
        self.pi = Actor_type(act_limit,**Actor_para)
        self.q1 = Q_type(**Q_para)
        self.q2 = Q_type(**Q_para)
        