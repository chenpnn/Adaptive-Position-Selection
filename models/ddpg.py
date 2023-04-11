import torch
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from models.model import *
from models.model import Memory, Actor, Critic, OUNoise

class DDPG():
    def __init__(self, 
                 env, 
                 hidden_size=128, 
                 actor_learning_rate=1e-4, 
                 critic_learning_rate=1e-3, 
                 gamma=0.99,  # discount factor 
                 tau=1e-2,  # update rate of target network, tau << 1
                 max_memory_size=50000
        ):
        self.counter = 0
        # Params
        self.num_states = env.num_macros  # 54
        self.num_actions = env.num_actions  # 2
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions).cuda()
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions).cuda()
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, 1).cuda()
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, 1).cuda()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).cuda()
        action = self.actor.forward(state)
        action = action.cpu().detach().numpy()[0]
        return action
    
    def update(self, batch_size):
        self.counter += 1
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).cuda()
        actions = torch.FloatTensor(actions).cuda()
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = torch.FloatTensor(next_states).cuda()
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)  
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q  # TD target
        critic_loss = self.critic_criterion(Qvals, Qprime)  # TD error

        # Actor loss
        policy_loss = - self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        if self.counter % 2 == 0:
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
