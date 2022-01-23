from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.RLalgo.base import OffPolicy

from utility.utils import *
from utility.RL import *

class dyna(object):
    '''
    class for dyna-style MBRL
    '''
    def __init__(self,RL: OffPolicy, env,phyloss_flag,
                 env_batch_size=200,real_buffer_size = 10000) -> None:
        super().__init__()
        self.RL = RL
        self.real_env = env
        self.test_env = deepcopy(self.real_env)
        self.optimizer = torch.optim.Adam(self.RL.env.parameters(),lr=1e-3)
        
        self.phyloss_flag = phyloss_flag
        self.env_batch_size = env_batch_size
        self.mseloss = nn.MSELoss()
        
        self.interact_env = self.RL.interact_env
        self.real_buffer = ReplayBuffer(obs_dim=self.RL.obs_dim, act_dim=self.RL.act_dim, size=real_buffer_size)

    def trainenv(self,buffer,max_train_iter,dataloss=True,phyloss=False,num_batch=20,printflag=False,trainflag=True):
        i=0
        data = bufferdata(buffer,self.env_batch_size*num_batch)
        loader = DataLoader(dataset=data, batch_size=self.env_batch_size,shuffle=True)
        while True:
            j=0
            losssum=0
            for o, o2, a in loader:
                j+=1
                self.optimizer.zero_grad()
                myo2=self.RL.env(o,a)
                if phyloss:
                    if not dataloss:
                        loss = self.RL.env.phyloss_f(o,myo2,a)
                    else: loss = self.mseloss(myo2,o2) + self.RL.env.phyloss_f(o,myo2,a)
                else:
                    loss = self.mseloss(myo2,o2)
                losssum+=loss.detach()
                if trainflag==True:
                    loss.backward()
                    self.optimizer.step()
            i+=1
            avgloss = (losssum/j).item()
            if avgloss < 1e-6 or i >= max_train_iter:
                if printflag:
                    print(
                        'Epoch: {}\tCase in Buffer: {}\tModel loss: {}'
                        .format(i,buffer.size,avgloss))
                break
        return avgloss

    def __call__(self, epoch, real_policy_action_after,fake_policy_action_after,update_after,update_every,
                RL_batch_size,test_every,num_test_episodes,RL_update_iter,RL_loop_per_epoch,
                noiselist,fake_env_loss_criteria=0.008,env_train_start_size=4000,env_num_batch=20,
                data_train_max_iter=20,phy_train_max_iter=50,mixed_train=False,usemodel=True, noref_flag=True):
        # torch.manual_seed(0)
        # np.random.seed(0)
        dataloss=1e6
        RL_trained_flag = False
        fake_o, fake_ep_ret, fake_ep_len = self.RL.env.reset(), 0, 0
        o, ep_ret, ep_len = self.real_env.reset(), 0, 0
        o=nptotorch(o)
        returnlist=[]
        sizelist=[]
        for i in range(epoch):
            for j in range(1,self.env_batch_size+1):
                with torch.no_grad():
                    if i*j>=real_policy_action_after: a = self.RL.ac.get_action(o, noiselist[0])
                    else: a = self.real_env.action_space.sample()
                    o, ep_ret, ep_len = self.interact_env(a, o, ep_ret, ep_len,self.real_env,
                        self.real_buffer,self.RL.max_ep_len,self.RL.ep_type,secondbuffer=self.RL.buffer,noref_flag=noref_flag)

            # if self.real_buffer.size>=env_train_start_size and usemodel: 
            #     # for _ in range(5):
            #     if self.phyloss_flag: self.trainenv(self.real_buffer,max_train_iter=phy_train_max_iter,dataloss=False,phyloss=True,num_batch=env_num_batch)
            #     else: self.trainenv(self.real_buffer,max_train_iter=data_train_max_iter,phyloss=False,num_batch=env_num_batch,printflag=False)
            #     dataloss=self.trainenv(self.real_buffer,max_train_iter=data_train_max_iter,phyloss=False,num_batch=env_num_batch,printflag=True)
            
            if self.real_buffer.size>=update_after and (self.real_buffer.size % update_every) == 0:
                for _ in range(RL_update_iter):
                    batch = self.real_buffer.sample_batch(RL_batch_size)
                    self.RL.update(data=batch)
                RL_trained_flag=True  

            if dataloss<fake_env_loss_criteria and usemodel:    
                for t in range(RL_loop_per_epoch):
                    # if self.RL.buffer.size >= update_after:
                        
                    # with torch.no_grad():
                    #     if self.RL.buffer.size >= fake_policy_action_after:
                    #         a = self.RL.ac.get_action(fake_o,noiselist[t])
                    #     else: a = self.RL.env.action_space.sample()
                    #     # if self.RL.buffer.size > fake_policy_action_after:policy = self.RL.ac.get_action
                    #     # else: policy=None
                    #     self.RL.env.eval()
                    #     # interact_fakeenvRef(self.real_buffer,self.RL.buffer,self.RL.env,self.env_batch_size,noiselist[self.RL.buffer.size],policy)
                    #     fake_o, fake_ep_ret, fake_ep_len = self.interact_env(a,
                    #         fake_o, fake_ep_ret, fake_ep_len,self.RL.env,self.RL.buffer,self.RL.max_ep_len,'d')
                    #     self.RL.env.train()

                    # if self.RL.buffer.size % update_every == 0 and self.RL.buffer.size>update_after:
                    if (t+1) % update_every == 0 and self.RL.buffer.size>update_after:
                        for _ in range(RL_update_iter):
                            batch = self.RL.buffer.sample_batch(RL_batch_size)
                            self.RL.update(data=batch)
                        RL_trained_flag=True  

                            
                    # if (t+1) % update_every == 0 and self.phyloss_flag and self.RL.buffer.size>env_train_start_size:
                    #     for _ in range(1):
                    #         self.trainenv(self.RL.buffer,phy_train_max_iter,dataloss=False,phyloss=True,num_batch=env_num_batch)
                    #     # self.trainenv(self.RL.buffer,max_train_iter=1,phyloss=False,num_batch=env_num_batch,printflag=True,trainflag=False)
                    # if (t+1) % update_every == 0 and (not self.phyloss_flag) and self.RL.buffer.size>env_train_start_size:
                    #     # for _ in range(int(10/self.RL.buffer.size*self.real_buffer.size)):    
                    #         self.trainenv(self.real_buffer,max_train_iter=phy_train_max_iter,num_batch=env_num_batch)
                self.RL.buffer.size,self.RL.buffer.ptr = self.real_buffer.size, self.real_buffer.ptr


            if (RL_trained_flag and i%test_every==0)or i==0:
                ret,max,min=test_RL(self.test_env,num_test_episodes,self.RL.max_ep_len, self.RL.ac,i=self.real_buffer.size)
                print('\nbuffer size: {}\t Retrun: {}'.format(self.real_buffer.size,ret))
                returnlist.append([ret,max,min])
                sizelist.append(self.real_buffer.size)
                returnhis = np.zeros([4,len(returnlist)])
                returnhis[1:] = np.array(returnlist).T
                returnhis[0]=np.array(sizelist)
                
                # torch.save(self.RL.env,'modelphy1'+str(i))
                # torch.save(self.real_buffer,'bufferphy1')
                np.save('free3',returnhis)

                        
            
                

    
    


if __name__=='__main__':
    import gym
    from src.NN import core,model
    from src.RLalgo import ddpg
    realenv = gym.make('CartPole-v0')
    fakeenv = model.fake_cartpole_env()
    RLinp = {"env":fakeenv,'actor_critic':core.MLPDDPG,'ac_kwargs':dict(hidden_sizes=[256]*2,act_space_type='d'),'ep_type':'finite'}
    mb = dyna(ddpg.DDPG,RLinp,realenv,False)
    mb(10,400,1000,50,100,1,10,400,1000)
