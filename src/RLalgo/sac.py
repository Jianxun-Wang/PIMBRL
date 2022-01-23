from copy import deepcopy
from typing import List
import itertools
import torch
from torch.optim import Adam
from NN.RL.stochastic import SAC_net
from utility.utils import *
from RLalgo.base import OffPolicy

class SAC(OffPolicy):
    
    def __init__(self, env, Actor, Q, act_space_type, a_kwargs:dict, q_kwargs:dict=None,
                replay_size=int(1e6), batch_size=100, interact_type = 's', max_ep_len = 1000, 
                ep_type='inf',  pi_lr=1e-3, q_lr=1e-3, gamma=0.99, 
                polyak=0.995, alpha=0.2, device=None):
        
        super(SAC, self).__init__(env, replay_size, batch_size, 
            interact_type, max_ep_len, ep_type, env.act_limit,device)

        a_kwargs = {'obs_dim':self.obs_dim, 
                    'act_dim':self.act_dim,
                    **a_kwargs}
        if q_kwargs == None: 
            q_kwargs={'obs_dim':self.obs_dim, 
                      'act_dim':self.act_dim,
                      **a_kwargs}
        self.ac:SAC_net = SAC_net(self.act_limit, 
                                  act_space_type, 
                                  Actor, a_kwargs,
                                  Q, q_kwargs)
        if device != None:
            self.ac=self.ac.to(device)
            self.ac.pi.act_len = self.ac.pi.act_len.to(device)
        self.ac_targ = deepcopy(self.ac)
        
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)
        
    
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        return loss_pi


    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True


        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

if __name__=='__main__':
    import torch.nn as nn
    from NN.RL.base import MLPQ
    from NN.RL.stochastic import GaussianMLPActor
    import gym
    RL = SAC(gym.make('CartPole-v0'),
              Actor=GaussianMLPActor, Q=MLPQ,
              a_kwargs=dict(activation=nn.ReLU,
                        hidden_sizes=[256]*2,
                        output_activation=nn.Tanh), 
                        act_space_type= 'd',)
    testenv = gym.make('CartPole-v0')
    para={'epoch':100000,'policy_action_after':1600,'update_after':1600,
    'update_every':100,'RL_update_iter':50,'batch_size':128,
    'test_every':2000,'num_test_episodes':100,
    'noiselist':torch.linspace(0.2,0.2,int(16e4))}
    RL(**para)