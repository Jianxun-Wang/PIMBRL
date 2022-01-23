import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from utility.utils import *


'''
envs should have following:
    variables:
        obs_dim: tuple or int 
        act_dim: tuple or int
        act_limit: tensor, shape act_dim*2
'''


class env_base(object):
    def __init__(self):
        super(env_base, self).__init__()
        self.action_space = action_space()
        self.act_dim = 0
        self.obs_dim = 0
        self.act_limit = 0
        # self.

    def step(self):
        raise NotImplementedError

    def step_p(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class action_space(object):
    def __init__(self, device='cpu'):
        super(action_space, self).__init__()
        self.device = device

    def sample(self):
        raise NotImplementedError


class pendulum(env_base):
    def __init__(self):
        super(pendulum, self).__init__()
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = 10.
        self.m = 1.
        self.l = 1.
        self.pi = torch.tensor(np.pi)
        self.action_space = pen_action_space()
        self.act_dim = 1
        self.obs_dim = 2
        self.act_limit = torch.tensor([[-2, 2], ])

    def step(self, a):
        a = clip_tensor(a, -self.max_torque, self.max_torque)
        costs = self.angle_normalize(
            self.state[0]) ** 2 + .1 * self.state[1] ** 2 + .001 * (a * a)

        newthdot = self.state[1] + (-3 * self.g / (2 * self.l) * torch.sin(
            self.state[0] + self.pi) + 3. / (self.m * self.l ** 2) * a) * self.dt
        newth = self.state[0] + newthdot * self.dt
        newthdot = clip_tensor(newthdot, -self.max_speed, self.max_speed)

        self.state = torch.tensor([newth, newthdot])
        return self.state, -costs, False, {}

    def angle_normalize(self, x):
        return (((x+self.pi) % (2*self.pi)) - self.pi)

    def reset(self):
        x = 2*torch.rand((2))-1
        self.state = torch.tensor([self.pi*x[0], x[1]])
        return self.state


class pen_action_space(action_space):
    def sample(self):
        return 4*torch.rand((1))-2


class burgers(env_base):
    '''
        meshx: 100
        max_t: 500
    '''

    def __init__(self):
        super(burgers, self).__init__()
        l = 2

        self.meshx = 150
        dx = l/self.meshx
        self.maxt = 60
        self.dx = 1/self.meshx
        self.dt = 0.001
        self.nu = 0.01

        x = torch.linspace(0, l-dx, self.meshx)
        self.f1 = torch.exp(-225*(x/l-.25)*(x/l-.25))
        self.f2 = torch.exp(-225*(x/l-.75)*(x/l-.75))
        self.init1 = 0.2*torch.exp(-25*(x/l-0.5)*(x/l-0.5))
        self.init2 = 0.2*torch.sin(4*math.pi*x/l)

        self.len = 0
        self.d = False

        self.loss = torch.nn.MSELoss()
        self.info = {}
        self.num_steps = 500
        ref = torch.arange(0, 30.5, self.dt*self.num_steps)
        self.ref = (0.05*torch.sin(np.pi/15*ref) +
                    0.5).reshape(-1, 1).repeat([1, self.meshx])

        self.action_space = burgers_action_space()
        self.act_dim = 2
        self.obs_dim = int(*x.shape)
        self.act_limit = torch.tensor([[-.025, .075], [-.025, .075]])

    def step(self, act):
        self.len += 1
        for _ in range(self.num_steps):
            self.pdestate = torch.cat(
                (self.pdestate[-2:], self.pdestate, self.pdestate[:2]))
            lapa = -1/12*self.pdestate[:-4]+4/3*self.pdestate[1:-3]-5/2 * \
                self.pdestate[2:-2]+4/3 * \
                self.pdestate[3:-1]-1/12*self.pdestate[4:]
            state2 = self.pdestate**2/2
            gradient = 0.5*state2[:-4]-2*state2[1:-3]+1.5*state2[2:-2]

            self.pdestate = self.pdestate[2:-2] + self.dt * (
                self.nu * lapa / self.dx**2 - gradient / self.dx
                + act[0]*self.f1 + act[1]*self.f2)

        self.state = self.pdestate-self.ref[self.len]
        # TODO when nan occurs, treat as a normal condition requires reset() and rew=-inf
        if torch.any(self.state.isnan()) == True:
            raise ValueError
        self.d = False if self.len < self.maxt else True

        rew = self.compute_rew()
        return self.state, rew, self.d, self.len

    def compute_rew(self):
        return -10*((self.state**2).mean())

    def reset(self):
        a = torch.rand(1)
        self.pdestate = a*self.init1 + (1-a)*self.init2 + 0.2
        self.state = self.pdestate - self.ref[0]
        self.d = False
        self.len = 0
        return self.state


class burgers_action_space(action_space):
    def sample(self, batch=None):
        if batch == None:
            shape = 2
        else:
            shape = (batch, 2)
        return 0.1*(torch.rand(shape))-0.025
        # return (0.1*torch.rand((2))-0.025)


@torch.jit.script
def RHS(u, lapa_c, lapa2_c, gradient_fc, gradient_bc, dx: float, dx2: float, dx4: float, f):
    u2 = u*u
    lapa = torch.matmul(lapa_c, u)
    lapa2 = torch.matmul(lapa2_c, u)
    gradient = torch.matmul(gradient_fc, u2)*(u < 0)\
        + torch.matmul(gradient_bc, u2)*(u >= 0)
    return -lapa2/dx4 - lapa/dx2 - gradient/2./dx + f, lapa, gradient


@torch.jit.script
def __calculate__(state, act, lapa_c, lapa2_c, gradient_fc, gradient_bc,
                  dt: float, dx: float, dx2: float, dx4: float, f0, f1, f2, f3, r, num_steps: int):

    f = act[0]*f0 + act[1]*f1 + act[2]*f2 + act[3]*f3
    for _ in range(num_steps):
        k1, lapa, gradient = RHS(
            state, lapa_c, lapa2_c, gradient_fc, gradient_bc, dx, dx2, dx4, f)
        k2, _, _ = RHS(state + dt*k1/2, lapa_c, lapa2_c,
                       gradient_fc, gradient_bc, dx, dx2, dx4, f)
        k3, _, _ = RHS(state + dt*k2/2, lapa_c, lapa2_c,
                       gradient_fc, gradient_bc, dx, dx2, dx4, f)
        k4, _, _ = RHS(state + dt*k3, lapa_c, lapa2_c,
                       gradient_fc, gradient_bc, dx, dx2, dx4, f)
        r += (lapa*lapa).mean() + (gradient*gradient).mean() + (state*f).mean()
        state = state + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    return state, -r/num_steps


class ks(env_base):
    '''
        meshx: 100
        max_t: 500
    '''

    def __init__(self, device='cuda:0'):
        super(ks, self).__init__()

        l = 8*math.pi

        self.meshx = 64
        dx = l/self.meshx
        self.maxt = 400
        self.dx = l/self.meshx
        self.dx2 = self.dx**2
        self.dx4 = self.dx**4
        self.dt = 0.001

        x = torch.linspace(0, l-dx, self.meshx).to(device)
        self.f0 = (torch.exp(-x**2/2)/math.sqrt(2*math.pi)).to(device)
        self.f1 = (torch.exp(-(x - 0.25*l)**2/2) /
                   math.sqrt(2*math.pi)).to(device)
        self.f2 = (torch.exp(-(x - 0.50*l)**2/2) /
                   math.sqrt(2*math.pi)).to(device)
        self.f3 = (torch.exp(-(x - 0.75*l)**2/2) /
                   math.sqrt(2*math.pi)).to(device)

        # self.init1 = torch.exp(-25*(x/l-0.5)**2)
        # self.init = torch.cos(x/16)*(1+torch.sin(x/16))
        self.init = torch.load('ks_init.tensor', map_location=device)
        # self.init2 = 1+torch.sin(1*math.pi*x/l)
        # self.init3 = torch.cos(1*math.pi*x/l)
        # self.init4 = torch.sin(2*math.pi*x/l)
        # self.init5 = torch.sin(8*math.pi*x/l)
        self.len = 0
        self.d = False

        # self.loss = torch.nn.MSELoss()
        self.info = {}
        self.num_steps = 250

        self.action_space = ks_action_space()
        self.act_dim = 4
        self.obs_dim = int(*x.shape)
        self.act_limit = torch.tensor(
            [[-.5, .5], [-.5, .5], [-.5, .5], [-.5, .5]])

        self.lapa_c = FD_Central_CoefficientMatrix(
            [1/90, -3/20, 3/2, -49/18], self.meshx, periodic=True)
        self.lapa2_c = FD_Central_CoefficientMatrix(
            [7/240, -2/5, 169/60, -122/15, 91/8], self.meshx, periodic=True)
        self.gradient_fc, self.gradient_bc = FD_upwind_CoefficientMatrix(
            [1/4, -4/3, 3, -4, 25/12], self.meshx, periodic=True)
        self.numpy = dict(f=torch.cat(
            (self.f0.unsqueeze(0), self.f1.unsqueeze(0),
             self.f2.unsqueeze(0), self.f3.unsqueeze(0)), dim=0).cpu().numpy(),
            init=self.init.cpu().numpy(),
            lapa_c=self.lapa_c.cpu().numpy(),
            lapa2_c=self.lapa2_c.cpu().numpy(),
            gradient_fc=self.gradient_fc.cpu().numpy(),
            gradient_bc=self.gradient_bc.cpu().numpy())

    def step(self, act):
        r = torch.zeros(1)
        self.state, rew = __calculate__(self.state, act, self.lapa_c, self.lapa2_c,
                                        self.gradient_fc, self.gradient_bc,
                                        self.dt, self.dx, self.dx2, self.dx4, self.f0,
                                        self.f1, self.f2, self.f3, r, self.num_steps)

        self.d = False if self.len < self.maxt else True
        self.len += 1
        return self.state, rew, self.d, self.info

    def step_p(self, state, act):
        state, rew = self.__calculate__(state, act)
        return state, rew

    def reset(self, shape=()):
        self.state = self.init[torch.randint(0, 200, size=shape)]
        self.d = False
        self.len = 0
        return self.state

    def test_reset(self, num_test=5):
        # return self.init[torch.multinomial(torch.ones(200), num_samples=num_test,
        #     replacement=False)].unsqueeze(2).cpu().numpy()
        return self.init[torch.linspace(0,199,num_test,dtype=torch.long)].unsqueeze(2).cpu().numpy()

    def RHS(self, u, f):
        u2 = u*u
        lapa = np.matmul(self.numpy['lapa_c'], u)
        lapa2 = np.matmul(self.numpy['lapa2_c'], u)
        gradient = np.matmul(self.numpy['gradient_fc'], u2)*(u < 0)\
            + np.matmul(self.numpy['gradient_bc'], u2)*(u >= 0)
        return -lapa2/self.dx4 - lapa/self.dx2 - gradient/2./self.dx + f, lapa, gradient

    def __calculate__(self, state, act):
        f = np.matmul(act, self.numpy['f'])
        f = np.expand_dims(f, axis=-1)
        r = 0
        for _ in range(self.num_steps):
            k1, lapa, gradient = self.RHS(state, f)
            k2, _, _ = self.RHS(state + self.dt*k1/2, f)
            k3, _, _ = self.RHS(state + self.dt*k2/2, f)
            k4, _, _ = self.RHS(state + self.dt*k3, f)
            r += (lapa*lapa).mean(axis=(1, 2)) + (gradient*gradient).mean(axis=(1, 2))\
                + (state*f).mean(axis=(1, 2))
            state = state + self.dt*(k1 + 2*k2 + 2*k3 + k4)/6
        return state, -r/self.num_steps


class ks_action_space(action_space):
    def sample(self, batch=None):
        if batch == None:
            shape = 4
        else:
            shape = (batch, 4)
        return (torch.rand(shape, device=self.device)-0.5)


if __name__ == '__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    import os

    torch.manual_seed(0)
    np.random.seed(0)
    import time
    result = []
    result2 = []
    env = ks(device='cpu')
    s = env.reset()
    a = torch.zeros(4)
    timelog = []
    startt = time.time()
    n = 2
    state = s.reshape(1, -1, 1).repeat([n, 1, 1]).cpu().numpy()
    act = torch.zeros((n, 4))
    for i in range(60):
        _, r, _, _ = env.step(a)
        # result.append(r)
        # state,r2=env.step_p(state,act.cpu().numpy())
        # result2.append(r2)
    tensortime = time.time()-startt
    print(tensortime)

    for n in [2, 4, 8, 16, 32, 64, 128, 256]:
        state = s.reshape(1, -1, 1).repeat([n, 1, 1]).cpu().numpy()
        act = torch.zeros((n, 4))
        start = time.time()
        for i in range(60):
            state, _ = env.step_p(state, act.cpu().numpy())
        state1 = torch.from_numpy(state)
        nptime = time.time()-start
        print(nptime)
        timelog.append(nptime/tensortime)
    plt.plot(timelog)
    plt.show()
    # x=torch.cat([i.unsqueeze(0) for i in result])
    # y=np.concatenate([j[0:1].squeeze(-1) for j in result2],axis=0)
    # plt.pcolormesh(x.detach().cpu())
    # plt.colorbar()
    # plt.show()
