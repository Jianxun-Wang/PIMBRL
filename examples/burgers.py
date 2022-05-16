from src.NN import model
from utility.RL import *
from envs import *
from ModelBase.dyna import dyna
from torch.utils.data import DataLoader

class phyburgers(dyna):
    def __init__(self, RL, env, phyloss_flag, env_batch_size, fake_buffer_size) -> None:
        super().__init__(RL, env, phyloss_flag, env_batch_size=env_batch_size, real_buffer_size=fake_buffer_size)
        self.real_buffer = BufferforRef(obs_dim=self.RL.obs_dim, act_dim=self.RL.act_dim, size=fake_buffer_size)
        self.RL.buffer = BufferforRef(obs_dim=self.RL.obs_dim, act_dim=self.RL.act_dim, size=self.RL.buffer.max_size)
    
    def trainenv(self,buffer,max_train_iter,dataloss=True,phyloss=False,num_batch=5,printflag=False,trainflag=True):
        i=0
        self.RL.env.train()
        for p in self.RL.env.parameters():
            p.requires_grad = True
        data = bufferdataref(buffer,self.env_batch_size*num_batch)
        loader = DataLoader(dataset=data, batch_size=self.env_batch_size,shuffle=True)
        
        while True:
            j=0
            losssum=0
            for o, o2, a, len in loader:
                j+=1
                self.optimizer.zero_grad()
                myo2=self.RL.env(o, a, len)
                # loss = (dataloss)*self.mseloss(myo2,o2) + (phyloss)*self.RL.env.phyloss_f(o, a)
                
                # loss.backward()
                # self.optimizer.step()
                if phyloss:
                    if not dataloss:
                        loss = self.RL.env.phyloss_f(o,a)
                    else: loss = self.mseloss(myo2,o2) + self.RL.env.phyloss_f(o,a)
                else:
                    loss = self.mseloss(myo2,o2)
                losssum+=loss.detach()
                if trainflag==True:
                    loss.backward()
                    self.optimizer.step()
            i+=1
            avgloss = (losssum/j).item()

            i+=1
            if avgloss<1e-6 or i>max_train_iter:
                if printflag:
                    print(
                        'Epoch: {}\tCase in Buffer: {}\tModel loss: {}'.format(
                            i,buffer.size,avgloss))
                break
        return avgloss


if __name__=='__main__':

    from NN.RL.base import MLPActor, MLPQ
    from RLalgo.td3 import TD3
    import torch.nn as nn
    import random
    import os

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    
    RL_batchsize=120
    env_batchsize=120
    realenv = burgers()
    fakeenv = model.fake_burgers_env(env_batchsize,ratio=1)
    #os.chdir('/home/lxy/store/RL/burgers/td3')
    
    RLinp = {"env":fakeenv,
            'Actor':MLPActor,
            "Q":MLPQ, 
            'act_space_type':'c',
            'a_kwargs':dict(activation=nn.ReLU,
                        hidden_sizes=[256]*2,
                        output_activation=nn.Tanh),
            'ep_type':'inf',
            'max_ep_len':60}
    RL = TD3(**RLinp) 
    mb = phyburgers(RL,realenv,False,env_batchsize,50000)
    
    mb(150,240,240,120,update_every=120 ,RL_batch_size=RL_batchsize,
        test_every=2, num_test_episodes=10, RL_loop_per_epoch=16,
        RL_update_iter = 50, env_train_start_size=120, 
        noiselist=torch.linspace(0.2,0.2,int(16e4)), 
        data_train_max_iter=50, mixed_train=False,
        fake_env_loss_criteria=0.02, usemodel=False,noref_flag=False)

