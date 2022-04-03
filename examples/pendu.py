import torch
import numpy as np
from src.ModelBase.dyna import *
from src.envs import pendulum
from src.NN.RL import model
from RLalgo.td3 import TD3

if __name__=='__main__':
    from src.NN.RL.base import MLPActor, MLPQ
    import os
    torch.manual_seed(10)
    np.random.seed(10)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    RL_batchsize=128
    env_batchsize=200
    realenv = pendulum()#lambda:gym.make('Pendulum-v0')
    fakeenv = model.fake_pendu_env(env_batchsize)
    os.chdir('/home/lxy/store/RL/pendulum/td3/')
    RLinp = {"env":fakeenv,
             'Actor':MLPActor,
             'Q': MLPQ, 
             'act_space_type':'c',
             'a_kwargs':dict(activation=nn.ReLU,
                        hidden_sizes=[256]*2,
                        output_activation=nn.Tanh),
            'ep_type':'inf', 'max_ep_len':200,
            'replay_size':int(5e5)}
    RL = TD3(**RLinp)
    mb = dyna(RL,realenv,False,env_batchsize,real_buffer_size=int(5e5))
    mb(1000,20000,20000,12000,update_every=200 ,RL_batch_size=RL_batchsize,test_every=4,
        num_test_episodes=100,RL_update_iter=50,RL_loop_per_epoch=1600,
        env_train_start_size=6000,noiselist=torch.linspace(0.1,0.0,int(16e4)),
        data_train_max_iter=50,mixed_train=False,
        fake_env_loss_criteria=1e7,usemodel=True)