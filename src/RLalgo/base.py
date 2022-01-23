from typing import Union
from copy import deepcopy
import torch.nn as nn
import src.utility.RL as RL
from src.envs import env_base
from utility.utils import *
from utility.RL import *
from NN.RL.base import ActorCritic


class OnPolicy(object):
    def __init__(self):
        super().__init__()

class OffPolicy(object):
    """
    The base class for off policy
    ---------
    - update

    """
    def __init__(self,
                env:Union[gym.wrappers.time_limit.TimeLimit, env_base, nn.Module], 
                replay_size:int=int(1e6), batch_size:int=100, 
                interact_type:str = 's', max_ep_len=200, 
                ep_type:str='inf', actlimit=None,device=None):

        super().__init__()
        self.ac:ActorCritic
        self.mseloss = nn.MSELoss()
        self.env = env
        self.batch_size = batch_size
        if isinstance(self.env,gym.wrappers.time_limit.TimeLimit):
            self.obs_dim = self.env.observation_space.shape[0]
            self.act_dim = 1
            self.act_limit = actlimit
        
        else:
            self.obs_dim = self.env.obs_dim
            self.act_dim = self.env.act_dim
            self.act_limit = self.env.act_limit
        self.buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size,device=device)
        self.interact_type = interact_type
        if type(interact_type)==str:
            self.interact_env = getattr(RL,'interact_env_'+interact_type)
        self.max_ep_len = max_ep_len
        self.ep_type = ep_type

    def update(self):
        """
            update RL policy
        """
        raise NotImplementedError

    def __call__(self, epoch,policy_action_after,update_after,update_every,
                 RL_update_iter,batch_size,test_every,num_test_episodes,noiselist):
        
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        self.num_test_episodes = num_test_episodes
        testenv = deepcopy(self.env)
        o = nptotorch(o)
        testcount = 0
        returnlist=[]
        sizelist=[]
        import time
        for i in range(epoch):
            with torch.no_grad():
                if i > policy_action_after:
                    a = self.ac.get_action(o, noiselist[0])
                else:
                    a = self.env.action_space.sample()
                o, ep_ret, ep_len = self.interact_env(a, o, ep_ret, ep_len,self.env,self.buffer,self.max_ep_len,self.ep_type)

            if i >= update_after and i % update_every == 0:
                start = time.time()
                for _ in range(RL_update_iter):
                    batch = self.buffer.sample_batch(batch_size)
                    self.update(data=batch)
                print(time.time()-start)
            if (i+1)%test_every==0 and i>=update_after or i==0:
                testcount+=1
                ret,max,min=test_RL(testenv,num_test_episodes,self.max_ep_len, self.ac,i=self.buffer.size)
                returnlist.append([ret,max,min])
                sizelist.append(self.buffer.size)
                print('RL buffer size: {}\tTest: {} th\t Retrun: {}'.format(self.buffer.size,testcount,ret))
        returnhis = np.zeros([4,len(returnlist)])
        returnhis[1:] = np.array(returnlist).T
        returnhis[0]=np.array(sizelist)
        np.save('free',returnhis)
    