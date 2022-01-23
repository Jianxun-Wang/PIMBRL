import gym
import torch
import numpy as np
from src.ModelBase.dyna import *
from src.NN import model
from RLalgo.td3 import TD3

if __name__=='__main__':
    import random
    import os
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    test_episodes=100
    RL_batchsize=128
    env_batchsize=200
    realenv = gym.make('CartPole-v0')
    fakeenv = model.fake_cartpole_env(env_batchsize)
    # os.chdir()
    RLinp = {"env":fakeenv, # the surogate environment defined above
            'net_type':'mlp',  # the type of network, MLP or CNN or anything else defined in src/NN/base.py
            'act_space_type':'d', # the type of action space, 'c' for continuous, 'd' for discrete
            'ac_kwargs':dict(activation=nn.ReLU,
                        hidden_sizes=[256]*2,
                        output_activation=nn.Tanh),# the hyperparameters of the network
            'ep_type':'inf', # the type of episode, 'inf' for infinite, 'finite' for finite (only inf is supported for now)
            'max_ep_len':400, # the maximum length of an episode
            'replay_size':int(5e5) # the max size of the replay buffer
            }
    RL = TD3(**RLinp)
    mb = dyna(RL,realenv,False,env_batchsize,real_buffer_size=int(5e5))
    mb(80,1000,1000,1000,update_every=100 ,RL_batch_size=RL_batchsize,test_every=4,
        num_test_episodes=test_episodes,RL_update_iter=50,RL_loop_per_epoch=4000,
        env_train_start_size=800,noiselist=torch.zeros(16000),mixed_train=False,
        data_train_max_iter=100, fake_env_loss_criteria=1e-4,env_num_batch=10)
#    RLinp = {"env":realenv,'actor_critic':core.MLPTD3,
#            'ac_kwargs':dict(hidden_sizes=[256]*2,act_space_type='d'),
#            'ep_type':'inf','max_ep_len':200,'replay_size':int(5e5),
#            'actlimit':torch.tensor([[0,1]])}
#    RL = rla.td3.td3(**RLinp)
#    para={'epoch':100000,'policy_action_after':1600,'update_after':1600,
#        'update_every':100,'RL_update_iter':50,'batch_size':RL_batchsize,
#        'test_every':2000,'num_test_episodes':test_episodes,
#        'noiselist':torch.linspace(0.2,0.2,int(16e4))}
#    RL(**para)

