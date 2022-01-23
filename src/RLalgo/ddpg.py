from copy import deepcopy
import torch
from torch.optim import Adam
import gym
from NN.RL.deterministic import DDPG_net
from utility.utils import *
from RLalgo.base import OffPolicy

'''
Two buffer, Two env
'''

class DDPG(OffPolicy):
    def __init__(self,env, Actor,Q, act_space_type,a_kwargs:dict, q_kwargs=None,
                replay_size=int(1e6), gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, 
                interact_type = 's', max_ep_len=200, ep_type='inf',actlimit=None):

        super(DDPG,self).__init__(env, replay_size, batch_size, 
            interact_type, max_ep_len, ep_type,actlimit)
        a_kwargs = {'obs_dim':self.obs_dim, 
                    'act_dim':self.act_dim,
                    **a_kwargs}
        if q_kwargs == None: 
            q_kwargs={'obs_dim':self.obs_dim, 
                      'act_dim':self.act_dim,
                      **a_kwargs}
        # Create actor-critic module and target networks
        self.ac = DDPG_net(self.act_limit, act_space_type,
                            Actor,
                            {'obs_dim':self.obs_dim, 
                             'act_dim':self.act_dim,
                             **a_kwargs},
                            Q,
                            {'obs_dim':self.obs_dim, 
                             'act_dim':self.act_dim,
                             **q_kwargs})
        # self.ac.float()
        self.ac_targ = deepcopy(self.ac)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)
        
        self.gamma = gamma
        self.polyak = polyak
        for p in self.ac_targ.parameters():
            p.requires_grad = False


    def compute_loss_q(self,data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = self.ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = self.mseloss(q,backup)
        return loss_q

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self,data):
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    
    def update(self,data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network 
        for p in self.ac.q.parameters():
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # in-place operations "mul_", "add_" to update target
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)



if __name__=='__main__':
    import torch.nn as nn
    from NN.RL.base import MLPActor, MLPQ
    RL = DDPG(gym.make('CartPole-v0'),
              Actor=MLPActor, Q=MLPQ,
              a_kwargs=dict(activation=nn.ReLU,
                        hidden_sizes=[256]*2,
                        output_activation=nn.Tanh), 
                        act_space_type= 'd',
                        actlimit=torch.tensor([[0,1]]))
    testenv = gym.make('CartPole-v0')
    para={'epoch':100000,'policy_action_after':1600,'update_after':1600,
    'update_every':100,'RL_update_iter':50,'batch_size':128,
    'test_every':2000,'num_test_episodes':100,
    'noiselist':torch.linspace(0.2,0.2,int(16e4))}
    RL(**para)