import torch
import torch.nn as nn
import numpy as np
import envs
from utility.utils import clip_tensor 

pi = np.pi

class action_space_d(object):
    def __init__(self, low,high,size):
        self.low = low
        self.high = high
        self.size = size
        
    def sample(self,batch=None):
        if batch==None:shape = (self.size,)
        else: shape=(batch,self.size)
        return torch.randint(low=self.low,high=self.high,size=shape)

class action_space_c(object):
    def __init__(self, low,high,size):
        self.low = low
        self.high = high
        self.size = size
        
    def sample(self,batch=None):
        if batch==None:shape = (self.size,)
        else: shape=(batch,self.size)
        return (self.high-self.low)*torch.rand(shape)+self.low
################################ CartPole #################################

class fake_cartpole_env(nn.Module):
    def __init__(self,batch_size):
        super(fake_cartpole_env, self).__init__()
        self.l1 = nn.Linear(5,12)
        self.l2 = nn.Linear(12,32)
        self.l3 = nn.Linear(32,64)
        self.l4 = nn.Linear(64,4)
        self.action_space = action_space_d(low=0,high=2,size=1)
        self.state = torch.zeros([4])
        self.d = False
        self.high = torch.tensor([2.4,float('inf'),0.209,float('inf')])
        self.low = torch.tensor([-2.4,float('-inf'),-0.209,float('-inf')])
        self.maxstep=200
        self.act = nn.ReLU()
        self.info = 'info is not provided for fake env'
        self.obs_dim=4
        self.act_dim=1
        self.act_limit=torch.tensor([[0,1],])
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02
        # self.zero = torch.zeros((1,4))
        self.loss_f = nn.MSELoss()


    def forward(self, state, act):
        # if not isinstance(act, torch.Tensor): act = torch.tensor([act])
        if state.dim()==1:state=state.unsqueeze(0)
        x = torch.cat((state,act),-1)
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        x = self.l4(x)
        return x

    def step(self,a):
        if isinstance(a,int):
            a=torch.tensor([[a]])
        else:a = a.unsqueeze(0)
        self.state = self.forward(self.state,a)
        self.d = ((self.state>self.high)+(self.state<self.low)).any().item()==True
        rew = 1 if not self.d else 0
        return self.state.squeeze(), rew, self.d, self.info

    def reset(self):
        self.state = -0.05 + 0.1*torch.rand(4)
        self.d=False
        return self.state

    def phyloss_f(self,s,s2,a):
        
        x, x_dot, theta, theta_dot = s[:,0:1],s[:,1:2],s[:,2:3],s[:,3:4]
        x2,x_dot2,theta2,theta_dot2 = s2[:,0:1],s2[:,1:2],s2[:,2:3],s2[:,3:4]
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        a = a*2-1
        force = self.force_mag*a
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc_c = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc_c = temp - self.polemass_length * thetaacc_c * costheta / self.total_mass
        
        x2c = x + self.tau * x_dot
        x_dot2c = x_dot + self.tau * xacc_c
        theta2c = theta + self.tau * theta_dot
        theta_dot2c = theta_dot + self.tau * thetaacc_c
        loss=torch.cat((x2-x2c,x_dot2-x_dot2c,theta2-theta2c, 
                            theta_dot2- theta_dot2c),-1)
        return self.loss_f(loss,torch.zeros_like(loss))

############################ End of CartPole #################################

############################ MountainCar #################################
class fake_mountaincar_env(nn.Module):
    def __init__(self,batch_size):
        super(fake_mountaincar_env, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3,12),
            nn.ReLU(),
            nn.Linear(12,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,2))
        
        self.d = False
        
        self.maxstep=200
        self.act = nn.ReLU()
        self.info = ''
        self.obs_dim=2
        self.zeros = torch.zeros((batch_size))
        self.act_dim=1
        self.act_limit=torch.tensor([[0,2],])
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.min_action = -1.
        self.max_action = 1.
        self.goal_position = 0.5
        self.goal_velocity = 0

        self.force = 0.0015
        self.gravity = 0.0025
        self.loss_f = nn.MSELoss()
        self.action_space = action_space_d(low=-1,high=1,size=self.act_dim)
        
    def forward(self, state, act):
        x = torch.cat((state,act),-1)
        return self.net(x)

    def step(self,a):
        if isinstance(a,float):
            a=torch.tensor([a])
        
        # a=min(max(a, self.min_action), self.max_action)
        a = a.unsqueeze(0)
        self.state = self.forward(self.state,a)
        velocity = self.state[0,1]
        position = self.state[0,0]
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0
        self.state = torch.tensor((position,velocity)).unsqueeze(0)
        self.d = (self.state[:,0]>=self.goal_position)*(self.state[:,1]>=self.goal_velocity)
        rew = -1#0
        # if self.d:
        #     rew = 100.0
        # rew -= a[0]*a[0] * 0.1
        return self.state[0], rew, self.d, self.info

    def reset(self):
        self.state = torch.rand((1,2))-0.6
        self.d=False
        return self.state[0]
    
    def phyloss_f(self,o,o2,a):
        
        vloss=o2[:,1]-o[:,1] - a.squeeze()*self.force + torch.cos(3*o[:,0])*self.gravity
        ploss=o2[:,0]-o[:,0] - o[:,1]
        return self.loss_f(ploss,0.)+self.loss_f(vloss,0.)
############################ End of MountainCar #################################

@torch.jit.script 
def cost(th,thdot,a,pi:float=pi):
    return (((th+pi) % (2*pi)) - pi) ** 2 + .1 * thdot *thdot + .001 * (a *a)

############################ Pendulem #################################
class fake_pendu_env(envs.pendulum,nn.Module):
    '''
        in this env, obs is different from state. 
        state is only for internal use (phyloss)
    '''
    def __init__(self,batch_size):
        super(fake_pendu_env, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3,15),
            nn.ReLU(),
            nn.Linear(15,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,2))
        
        self.d = False
        self.high = torch.tensor([-1,-1,-8])
        self.low = torch.tensor([1,1,8])
        self.maxstep=200
        self.act = nn.ReLU()
        self.info = 'info is not provided for fake env'
        
        self.zeros = torch.zeros((batch_size))
        self.loss_f = nn.MSELoss()
        self.maxT = 2*torch.ones([batch_size,1])
        self.maxspeed = 8*torch.ones([batch_size,1])
       
    def forward(self, obs, act, maxT=None,maxspeed=None):
        if maxT==None: maxT = self.maxT
        if maxspeed==None: maxspeed=self.maxspeed
        act = clip_tensor(act,-maxT, maxT)
        state = torch.cat((obs,act),-1)
        x = self.net(state)
        xdot = clip_tensor(x[:,1:],-maxspeed,maxspeed)
        return torch.cat((x[:,0:1],xdot),1)

    def step(self,a):
        a=a.unsqueeze(0)
        self.state=self.forward(self.state,a,self.maxT[0:1],self.maxspeed[0:1])
        return self.state[0], -cost(self.state[:,0],self.state[:,1],a), False, self.info

    def reset(self):
        x = torch.rand((1,2))
        self.state = torch.cat((pi*x[:,0:1],x[:,1:]),1)
        return self.state[0]

    def phyloss_f(self,o,o2,a):
        a=a.squeeze()
        ids = ((o2[:,1]<self.maxspeed[:,0]) * (o2[:,1]>-self.maxspeed[:,0])).nonzero().squeeze()
        loss1=(o2[ids,1] - o[ids,1] - (-3 * self.g / (2 * self.l) * torch.sin(o[ids,0] + pi) 
              + 3. / (self.m * self.l ** 2) * a[ids]) * self.dt)
        loss2=o2[ids,0] - o[ids,0] - o[ids,1] * self.dt
        return self.loss_f(loss1,self.zeros[ids]) + self.loss_f(loss2,self.zeros[ids])
############################ End of Pendulem #################################

############################ Burger #################################

class fake_burgers_env(envs.burgers, nn.Module):
    def __init__(self,batch_size,ratio=50):
        super(fake_burgers_env,self).__init__()
        self.encoder = nn.Sequential(
                            nn.Conv1d(1,8,6,stride=3),
                            nn.ReLU(),
                            nn.Conv1d(8,16,7,stride=3),
                            nn.ReLU(),
                            nn.Conv1d(16,32,5,stride=2),
                            nn.ReLU(),
                            nn.Conv1d(32,48,6))
        
        self.l = nn.Sequential(nn.Linear(48,24),nn.ReLU(),nn.Linear(24,10))
        self.lstm = nn.LSTM(2,10,1,batch_first=True)
        self.decoder = nn.Sequential(
                            nn.Linear(10,36),
                            nn.ReLU(),
                            nn.Linear(36,72),
                            nn.ReLU(),
                            nn.Linear(72,150))
        
        self.dt = self.dt*ratio
        self.info = 'No info for fake env'
        self.seq_len = int(self.num_steps/ratio)
        self.loss = nn.MSELoss()
        self.f1 = self.f1.reshape(1,1,-1).repeat([batch_size,self.seq_len,1])
        self.f2 = self.f2.reshape(1,1,-1).repeat([batch_size,self.seq_len,1])
        self.zero = torch.zeros((batch_size,self.seq_len,self.meshx))
        self.mseloss = nn.MSELoss()
        self.maxt=self.maxt*torch.ones(batch_size)

    def forward(self, state, act, len):
        pdestate = state + self.ref[len-1]
        x = self.encoder(pdestate.unsqueeze(1))
        x = self.l(x.squeeze(-1)).unsqueeze(1)
        h,c = x.permute(1,0,2),x.permute(1,0,2)
        act = act.unsqueeze(1).repeat([1,self.seq_len,1])
        x,(h,c) = self.lstm(act,(h,c))
        self.result = self.decoder(x)
        return self.result[:,-1] - self.ref[len]


    def step(self,o,a,len):
        d = (len<self.maxt)
        filterid = d.nonzero().squeeze()
        len = len[filterid]+1
        o,a = o[filterid],a[filterid]
        o2 = self.forward(o,a,len)
        d = (len>=self.maxt[filterid])
        rew = -10*((o2**2).mean(axis=1))
        return o,a,o2, rew, d, len, filterid.shape[0]


    def reset(self):
        tmp=super().reset()
        self.state = self.state.unsqueeze(0)
        # self.pdestate = self.pdestate.unsqueeze(0)
        return tmp

    def phyloss_f(self,o,a):
        
        a = a.unsqueeze(1).repeat([1,self.seq_len,1])
        x = torch.cat((self.result[:,:,-2:],self.result,self.result[:,:,:2]),2)
        lapa = -1/12*x[:,:,:-4]+4/3*x[:,:,1:-3]-5/2*x[:,:,2:-2]+4/3*x[:,:,3:-1]-1/12*x[:,:,4:]
        state2 = x**2/2
        gradient = 0.5*state2[:,:,:-4]-2*state2[:,:,1:-3]+1.5*state2[:,:,2:-2]
        x_grad = self.nu * lapa / self.dx**2 - gradient / self.dx \
                        +a[:,:,:1]*self.f1 + a[:,:,1:]*self.f2
        x = torch.cat((o.unsqueeze(1),x[:,:,2:-2]),1)
        residual = (x[:,1:] - x[:,:-1])/self.dt - x_grad

        return self.mseloss(residual,self.zero)

        
############################ End of Burger #################################
@torch.jit.script
def __rew__(lapa_c,gradient_fc,gradient_bc,u,a,f0,f1,f2,f3):
    ur = u.unsqueeze(3)
    u2 = ur*ur
    lapa = torch.matmul(lapa_c,ur).squeeze()
    gradient = torch.matmul(gradient_fc,u2).squeeze()*(u<0)\
                +torch.matmul(gradient_bc,u2).squeeze()*(u>=0)
    f = a[:,:,0:1]*f0 + a[:,:,1:2]*f1 + a[:,:,2:3]*f2 + a[:,:,3:]*f3
    return -(lapa*lapa).mean(dim=(1,2)) - (gradient*gradient).mean(dim=(1,2)) - (u*f).mean(dim=(1,2))

@torch.jit.script 
def RHS(u,lapa_c,lapa2_c,gradient_fc,gradient_bc,dx:float,dx2:float,dx4:float,f):
    u1=u.unsqueeze(3)
    state2=u1*u1
    lapa=torch.matmul(lapa_c,u1).squeeze()
    lapa2=torch.matmul(lapa2_c,u1).squeeze()
    gradient=torch.matmul(gradient_fc,state2).squeeze()*(u<0)\
                +torch.matmul(gradient_bc,state2).squeeze()*(u>=0)
    return -lapa2/dx4 - lapa/dx2 - gradient/2./dx + f

@torch.jit.script
def __loss__(u,u1,a,lapa_c,lapa2_c,gradient_fc,gradient_bc,
                 dx:float,dx2:float,dx4:float,dt:float,f0,f1,f2,f3):
    f = a[:,:,0:1]*f0 + a[:,:,1:2]*f1 + a[:,:,2:3]*f2 + a[:,:,3:]*f3
    k1 = RHS(u,lapa_c,lapa2_c,gradient_fc,gradient_bc,dx,dx2,dx4,f)
    k2 = RHS(u + dt*k1/2,lapa_c,lapa2_c,gradient_fc,gradient_bc,dx,dx2,dx4,f)
    k3 = RHS(u + dt*k2/2,lapa_c,lapa2_c,gradient_fc,gradient_bc,dx,dx2,dx4,f)
    k4 = RHS(u + dt*k3,lapa_c,lapa2_c,gradient_fc,gradient_bc,dx,dx2,dx4,f)
    
    return u1-u - dt*(k1 + 2*k2 + 2*k3 + k4)/6

############################ KS #################################

class fake_ks_env(envs.ks,nn.Module):
    def __init__(self,batch_size,ratio=5,forward_size=1):
        super(fake_ks_env,self).__init__()
        self.encoder = nn.Sequential(
                            nn.Conv1d(1,8,7,stride=3),
                            nn.ReLU(),
                            nn.Conv1d(8,16,6,stride=2),
                            nn.ReLU(),
                            nn.Conv1d(16,32,5,stride=1),
                            nn.ReLU(),
                            nn.Conv1d(32,48,4))
        
        self.l = nn.Sequential(nn.Linear(48,24),nn.ReLU(),nn.Linear(24,12))
        self.lstm = nn.LSTM(4,12,1,batch_first=True)
        self.decoder = nn.Sequential(
                            nn.Linear(12,24),
                            nn.ReLU(),
                            nn.Linear(24,48),
                            nn.ReLU(),
                            nn.Linear(48,64))
        
        self.dt = self.dt*ratio
        self.info = 'No info for fake env'
        self.seq_len = int(self.num_steps/ratio)
        self.loss = nn.MSELoss()
        self.f0f = self.f0.reshape(1,1,-1).repeat([forward_size,self.seq_len,1])
        self.f1f = self.f1.reshape(1,1,-1).repeat([forward_size,self.seq_len,1])
        self.f2f = self.f2.reshape(1,1,-1).repeat([forward_size,self.seq_len,1])
        self.f3f = self.f3.reshape(1,1,-1).repeat([forward_size,self.seq_len,1])
        self.f0 = self.f0.reshape(1,1,-1).repeat([batch_size,self.seq_len,1])
        self.f1 = self.f1.reshape(1,1,-1).repeat([batch_size,self.seq_len,1])
        self.f2 = self.f2.reshape(1,1,-1).repeat([batch_size,self.seq_len,1])
        self.f3 = self.f3.reshape(1,1,-1).repeat([batch_size,self.seq_len,1])
        self.zero = torch.zeros((batch_size,self.seq_len,self.meshx))
        self.mseloss = nn.MSELoss()
        self.forward_size = forward_size

    def forward(self, state, act):
        
        x = self.encoder(state.unsqueeze(1))
        x = self.l(x.squeeze(-1)).unsqueeze(0)
        h,c = x,x
        act = act.unsqueeze(1).repeat([1,self.seq_len,1])
        x,(h,c) = self.lstm(act,(h,c))
        self.result = self.decoder(x)
        return self.result[:,-1]


    def step(self,state,a):
        # a=a.unsqueeze(0)
        state = self.forward(state,a)
        
        # self.d = False if self.len<self.maxt else True
        # self.len +=1
        rew=__rew__(self.lapa_c, self.gradient_fc,self.gradient_bc,
                    self.result,a.unsqueeze(1).repeat([1,self.seq_len,1]),
                    self.f0f,self.f1f,self.f2f,self.f3f)
        return state, rew, False, self.info



    def phyloss_f(self,o,myo2,a):
        
        a = a.unsqueeze(1).repeat([1,self.seq_len,1])
        # result = self.result.unsqueeze(3)
        x = torch.cat((o.unsqueeze(1),self.result),1)
        residual = __loss__(x[:,:-1],self.result, a,  self.lapa_c,self.lapa2_c,
                                self.gradient_fc,self.gradient_bc,self.dx,self.dx2,
                                self.dx4,self.dt,self.f0,self.f1,self.f2,self.f3)
        return self.mseloss(residual,self.zero)
############################ End of KS #################################