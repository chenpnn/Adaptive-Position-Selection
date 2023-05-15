
import random
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size_list, output_size):
        super(Critic, self).__init__()
        linear_list = [nn.Linear(input_size, hidden_size_list[0]), nn.ReLU()]
        for i in range(len(hidden_size_list)):
            if i < len(hidden_size_list) - 1:
                linear_list.append(nn.Linear(hidden_size_list[i], hidden_size_list[i+1]))
                linear_list.append(nn.ReLU())
            else:
                linear_list.append(nn.Linear(hidden_size_list[i], output_size))
        self.linears = nn.Sequential(*linear_list)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = self.linears(x)
        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size_list, output_size):
        super(Actor, self).__init__()
        linear_list = [nn.Linear(input_size, hidden_size_list[0]), nn.ReLU()]
        for i in range(len(hidden_size_list)):
            if i < len(hidden_size_list) - 1:
                linear_list.append(nn.Linear(hidden_size_list[i], hidden_size_list[i+1]))
                linear_list.append(nn.ReLU())
            else:
                pass
                # linear_list.append(nn.Linear(hidden_size_list[i], output_size))
                # linear_list.append(nn.Tanh())
        self.common_linear = nn.Sequential(*linear_list)
        self.m = hidden_size_list[-1] // 2
        self.n = hidden_size_list[-1] - self.m

        self.linear1 = nn.Sequential(
            nn.Linear(self.m, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(self.n, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh(),
        )
        
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = self.common_linear(state)
        x1, x2 = x[:, :self.m], x[:, self.n:]
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)

        # x1 = nn.ReLU()(x[:, 0])  # scale
        # x2 = 3 * nn.Tanh()(x[:, 1])  # threshold
        out = torch.cat([x1.reshape(-1, 1), 
                       x2.reshape(-1, 1)], 1)
        return out


# class Actor(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Actor, self).__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, hidden_size)
#         self.linear3 = nn.Linear(hidden_size, output_size)
        
#     def forward(self, state):
#         """
#         Param state is a torch tensor
#         """
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         x = torch.tanh(self.linear3(x))

#         return x

class OUNoise(object):
    def __init__(self, 
                 action_dim,
                 low=-100,
                 high=100,
                 mu=0.0, 
                 theta=0.15, 
                 max_sigma=0.3, 
                 min_sigma=0.3, 
                 decay_period=100000
        ):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.low          = low
        self.high         = high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)





if __name__=='__main__':
    MAX_SIZE = 10000
    NUM_STATES, NUM_ACTIONS = 30, 3
    HIDDEN_SIZE = 128
    buffer = Memory(max_size=10000)
    actor = Actor(input_size=NUM_STATES, hidden_size_list=[128, 128], output_size=NUM_ACTIONS)
    actor_target = Actor(input_size=NUM_STATES, hidden_size_list=[128, 128], output_size=NUM_ACTIONS)
    critic = Critic(input_size=NUM_STATES+NUM_ACTIONS, hidden_size_list=[128, 128], output_size=NUM_ACTIONS)
    critic_target = Critic(input_size=NUM_STATES+NUM_ACTIONS, hidden_size_list=[128, 128], output_size=NUM_ACTIONS)

    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)

    actor = Actor(100, [64, 128], 2)
    x = torch.rand(32, 100)
    out = actor(x)


