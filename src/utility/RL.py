r"""RL related tools

- ReplayBuffer
- interact_env
- interact_fakeenv
- test_RL
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import gym
from .utils import combined_shape, nptotorch

class ReplayBuffer(object):
    """
    First In First Out experience replay buffer agents.
    """

    def __init__(self, obs_dim, act_dim, size, device=None):
        super(ReplayBuffer,self).__init__()
        self.obs_buf = torch.zeros(combined_shape(size, obs_dim), dtype=torch.float)
        self.obs2_buf = torch.zeros(combined_shape(size, obs_dim), dtype=torch.float)
        self.act_buf = torch.zeros(combined_shape(size, act_dim), dtype=torch.float)
        self.rew_buf = torch.zeros(size, dtype=torch.float)
        self.done_buf = torch.zeros(size, dtype=torch.float)
        if device is not None:
            self.obs_buf = self.obs_buf.to(device)
            self.obs2_buf = self.obs2_buf.to(device)
            self.act_buf = self.act_buf.to(device)
            self.rew_buf = self.rew_buf.to(device)
            self.done_buf = self.done_buf.to(device)
                
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done,store_size=1):
        self.obs_buf[self.ptr:self.ptr+store_size] = obs
        self.obs2_buf[self.ptr:self.ptr+store_size] = next_obs
        self.act_buf[self.ptr:self.ptr+store_size] = act
        self.rew_buf[self.ptr:self.ptr+store_size] = rew
        self.done_buf[self.ptr:self.ptr+store_size] = done
        self.ptr = (self.ptr+store_size) % self.max_size
        self.size = min(self.size+store_size, self.max_size)
    
    def sample_batch(self, batch_size=32,start=0,end=int(1e8)):
        idxs = torch.randint(start, min(self.size,end), size=(batch_size,))
        return dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])

class BufferforRef(ReplayBuffer):
    '''
    ReplayBuffer for environments with a reference (e.g. Burgers)
    '''
    def __init__(self, obs_dim, act_dim, size):
        super().__init__(obs_dim, act_dim, size)
        self.len = torch.zeros(size,dtype=torch.long)
    
    def store(self, obs, act, rew, next_obs, done,len,store_size=1):
        self.len[self.ptr:self.ptr+store_size] = len
        super().store(obs, act, rew, next_obs, done,store_size)
    
    def sample_batch(self, batch_size=32,start=0,end=int(1e8)):
        idxs = torch.randint(start, min(self.size,end), size=(batch_size,))
        return dict(obs=self.obs_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs],
                    len=self.len[idxs])
        
        
class bufferdata(Dataset):
    def __init__(self,buffer,size=float('inf')):
        super(bufferdata,self).__init__()
        self.size = min(size,buffer.size)
        self.idxs = torch.randint(0, buffer.size, size=(self.size,))
        self.obs = buffer.obs_buf[self.idxs]
        self.obs2 = buffer.obs2_buf[self.idxs]#[:self.size]
        self.act = buffer.act_buf[self.idxs]#[:self.size]
        self.rew = buffer.rew_buf[self.idxs]
        
    
    def __getitem__(self, index):
        return self.obs[index], self.obs2[index], self.act[index]
    
    def __len__(self):
        return self.size

class bufferdataref(bufferdata):
    def __init__(self, buffer: BufferforRef,size=None):
        super().__init__(buffer,size=size)
        self.len = buffer.len[self.idxs]

    def __getitem__(self, index):
        return *super().__getitem__(index), self.len[index]
     

def test_RL(env,num_test_episode,max_len,RLNN,parallel=False,i=0):
    returnlist=[]
    # logger = np.zeros((60,150))
    # rwlogger=np.zeros(60)
    # actlogger=np.zeros((60,2))
    RLNN.eval()
    if parallel:
        o, ep_ret, ep_len = env.test_reset(), 0, 0
        while(ep_len < max_len):
            a = RLNN.get_action(torch.Tensor(o.squeeze(-1)), 0)
            o, r = env.step_p(o,a.cpu().numpy())
            ep_ret = r + ep_ret
            ep_len += 1
        returnlist=ep_ret
        return_=np.array(returnlist,dtype=np.float)
    else:
        for j in range(num_test_episode):
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            o = nptotorch(o)
            while not(d or (ep_len >= max_len)):
                a = RLNN.get_action(o, 0)
                if isinstance(env, gym.wrappers.time_limit.TimeLimit):
                    if isinstance(env.action_space,gym.spaces.discrete.Discrete):
                        if isinstance(a,torch.Tensor):a=int(a)
                    else:a=a.numpy()
                o, r, d, _ = env.step(a)
                o = nptotorch(o)
                ep_ret = r + ep_ret
                ep_len += 1
            #     if j==0: 
            #         logger[ep_len-1]=o.cpu()
            #         rwlogger[ep_len-1]=r.cpu()
            #         actlogger[ep_len-1]=a.cpu()
            # if j==0:
            #     np.save('act'+str(i),actlogger)
            #     np.save('state'+str(i),logger)
            #     np.save('rew'+str(i),rwlogger)
            #     plt.pcolormesh(logger)
            #     plt.colorbar()
            #     plt.savefig(str(i))
            #     plt.close()
            #     plt.plot(actlogger)
            #     plt.savefig('act'+str(i))
            #     plt.close()
            returnlist.append(ep_ret)
            return_=torch.tensor(returnlist,dtype=torch.float)
    mean,max,min=return_.mean().item(),return_.max().item(),return_.min().item()
    RLNN.train()
    return mean,max,min


def interact_env_s(a, o, ep_ret, ep_len, env, buffer, max_len,ep_type,noref_flag=True,secondbuffer=None):
    '''
    Sequentially interact with env and save data pair to buffer
    '''
    # Step the env
    if isinstance(env, gym.wrappers.time_limit.TimeLimit):
        if isinstance(env.action_space,gym.spaces.discrete.Discrete):
            if isinstance(a,torch.Tensor):a=int(a)
        if isinstance(a,torch.Tensor):a=a.numpy()
    # else: a=nptotorch(a)
    o2, r, d, len = env.step(a)
    o2 = nptotorch(o2)
    ep_ret += r
    ep_len += 1

    if ep_type=='finite':
        d = False if ep_len==max_len else d
    if isinstance(a,np.ndarray):a=nptotorch(a)
    # Store experience to replay buffer
    if noref_flag:
        if secondbuffer!=None:
            secondbuffer.store(o,a,r,o2,d)
        buffer.store(o, a, r, o2, d)
    else: 
        if secondbuffer!=None:
            secondbuffer.store(o,a,r,o2,d,len)
        buffer.store(o, a, r, o2, d,len)
    
    o = o2

    # End of trajectory handling
    if d or (ep_len == max_len):
        o, ep_ret, ep_len = env.reset(), 0, 0
    return nptotorch(o), ep_ret, ep_len


def interact_env_p(a, o, ep_ret, ep_len, env, 
                   buffer, max_len,parallel_size,noref_flag=True):
    '''
    Parallel variant of interact_env_s, should be used for all continuious trajactories
    policy action delay should be divided by the batch_size

    Parallel in dimension of batches, but iterate over time steps 
    '''
    raise NotImplementedError


def interact_fakeenv(source_buffer:ReplayBuffer,save_buffer:ReplayBuffer,
        fake_env:torch.nn.Module,batch_size:int, noise_scale,policy=None,
        end=int(1e8)):
    data = source_buffer.sample_batch(batch_size,end=end)
    o,d = data['obs2'],data['done']
    if policy==None: a = fake_env.action_space.sample(batch_size) 
    else: a = policy(o,noise_scale)
    o2,r,_,_ = fake_env.step(o,a)
    save_buffer.store(o,a,r,o2,d,store_size=batch_size)

def step_fakeenv(source_buffer:ReplayBuffer,fake_env:torch.nn.Module,
                batch_size:int, noise_scale,policy=None):
    data = source_buffer.sample_batch(batch_size)
    o,d = data['obs2'],data['done']
    if policy==None: a = fake_env.action_space.sample(batch_size) 
    else: a = policy(o, noise_scale)
    o2,r,_,_ = fake_env.step(o,a)
    return dict(obs=o, obs2=o2, act=a, rew=r, done=d)
    

def interact_fakeenvRef(real_buffer:ReplayBuffer,fake_buffer:ReplayBuffer,
        fake_env:torch.nn.Module,batch_size:int, noise_scale,policy=None,
        end=int(1e8)):
    data = real_buffer.sample_batch(batch_size,end=end)
    o,len = data['obs2'],data['len']
    if policy==None: a = fake_env.action_space.sample(batch_size) 
    else: a = policy(o,noise_scale)
    o,a,o2,r,d,len,batch_size = fake_env.step(o,a,len)
    fake_buffer.store(o,a,r,o2,d,len,store_size=batch_size)