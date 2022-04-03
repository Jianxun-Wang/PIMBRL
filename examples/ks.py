import os
import random
import torch
import numpy as np
from src.ModelBase.dynav2 import *
from src.envs import *
from src.NN import model
from RLalgo.td3 import TD3

if __name__=='__main__':
    from src.NN.RL.base import MLPActor, MLPQ
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # set GPU device or use CPU
    device=torch.device('cuda:0')
    torch.cuda.set_device(device)

    # disable TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # set batch size
    RL_batchsize=128
    env_batchsize=400

    # set environment, OpenAI Gym is supported, you can define your own environment in envs.py
    realenv = ks(device=device)

    # set the model(support both physicas-informed and non-physicas-informed)
    fakeenv = model.fake_ks_env(env_batchsize,ratio=1,forward_size=400)

    # change the directory to save results
    # os.chdir(os.path.expanduser("~")+'/store/RL/ks/td3/free3')

    # define the RL network and other hyperparameters
    RLinp = {"env":fakeenv, # the surogate environment defined above
            'Actor':MLPActor,  # the type of policy network, src/NN/RL
            'Q': MLPQ, # the type of value function network, src/NN/RL
            'act_space_type':'c', # the type of action space, 'c' for continuous, 'd' for discrete
            'a_kwargs':dict(activation=nn.ReLU,
                        hidden_sizes=[256]*2,
                        output_activation=nn.Tanh),# the hyperparameters of the network
            'ep_type':'inf', # the type of episode, 'inf' for infinite, 'finite' for finite (only inf is supported for now)
            'max_ep_len':400, # the maximum length of an episode
            'gamma':0.977, # the discount factor
            'replay_size':int(5e5) # the max size of the replay buffer
            }
    RL = TD3(**RLinp)
    
    # define the dyna hyperparameters
    mb = dyna(RL,
            realenv,
            False, # whether to use the physicas-informed model
            env_batchsize,
            real_buffer_size=int(5e5))
    
    mb(epoch=1600000,
        real_policy_action_after = 16000,
        fake_policy_action_after = 16000,
        update_after = 12000, 
        RL_batch_size=RL_batchsize, 
        test_every=3, 
        num_test_episodes=200, # the number of episodes used to test the performance of the RL agent
        RL_update_iter=50, # the number of RL update for each iteration
        noiselist=torch.linspace(0.2,0.2,int(16e5)), # artificial noise added to actions
        phy_train_max_iter=21, 
        fake_env_loss_criteria=0.01, # the criteria for the beginning of using the fake environment
        env_train_start_size=6000,
        fake_len=3,
        usemodel=True, # use model-based RL or model-free RL
        RL_loop=10,
        refresh_RL_buffer_interval=3)