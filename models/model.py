
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
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
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

# https://medium.com/analytics-vidhya/a-simple-reinforcement-learning-environment-from-scratch-72c37bb44843

# 三类数据集：宏观特征(T*M)、单因子(T*N)和股票收益率(T*N)
# T：交易日个数、M：宏观特征个数、N：股票只数
# 给定一个日期t，提取当日宏观特征向量m_t，因子向量f_t, 股票收益率向量ret_t
# 输入一个动作a_t=(scale_t, threshold_t)，
# 计算得到一个权重， w_t = tanh(scale_t * max(f_t - threshold_t, 0))
# 计算得到当日奖励 r_t = ret_t @ w_t

# 环境需要一个idx，来标记当前日期，初始时为0，滑动到最后时一轮游戏结束
# class TradeEnv(object):
#     def __init__(self, 
#                  path='./data/processed/',
#                  start_date=None,
#                  end_date=None,
#         ):
#         self.macro_data = pd.read_csv(path+'macro_data.csv')
#         self.factor_data = None
#         self.stock_data = None
#         self.dates = None

#         self.num_actions = 2
#         self.num_states = self.macro_data.shape[1] - 1
#         self.done = False
#         self.idx=0
#         # self.state_observation = [self.x, self.y]

#     def _to_weight(self, factor, action):
#         # the larger the scale, more concentrated the weights
#         scale, threshold = action
#         factor = np.array(factor)
#         w = np.tanh(scale * np.maximum(factor-threshold, 0))
#         w /= np.sum(w)
#         return w

#     # reset the agent when an episode begins    
#     def reset(self):
#         self.idx = 0
#         self.done = False

#     # Agent takes the step, i.e. take action to interact with the environment
#     def step(self, action):
        
#         factor = None
#         self.weight = self._to_weight(factor, action)
#         reward = self.get_reward()
#         return np.array(self.state_observation), self.reward, self.done
    
#     # Action reward given to the agent
#     def get_reward(self):
#         ret = None
#         reward = ret @ self.weight
#         return reward
    
#     # Actual action that agent takes.
#     def take_action(self):
#         self.idx += 1  # jump to the next date







if __name__=='__main__':
    MAX_SIZE = 10000
    NUM_STATES, NUM_ACTIONS = 30, 3
    HIDDEN_SIZE = 128
    buffer = Memory(max_size=10000)
    actor = Actor(input_size=NUM_STATES, hidden_size=HIDDEN_SIZE, output_size=NUM_ACTIONS)
    actor_target = Actor(input_size=NUM_STATES, hidden_size=HIDDEN_SIZE, output_size=NUM_ACTIONS)
    critic = Critic(input_size=NUM_STATES+NUM_ACTIONS, hidden_size=HIDDEN_SIZE, output_size=NUM_ACTIONS)
    critic_target = Critic(input_size=NUM_STATES+NUM_ACTIONS, hidden_size=HIDDEN_SIZE, output_size=NUM_ACTIONS)

    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)