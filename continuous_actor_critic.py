# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:05:38 2020

@author: speechlab
"""

import torch
import torch.nn as nn
from prepare_embeddings import ProcessFrame96
import gym
from torch.distributions.normal import Normal
import math
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(4, 4), stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(4, 4), stride=2)
        
        self.dense = nn.Linear(in_features=40*14*14, out_features=100)
        self.mu = nn.Linear(in_features=100, out_features=3)
        self.var = nn.Linear(in_features=100, out_features=3)
        self.critic = nn.Linear(in_features=100, out_features=1)
        
    
    def forward(self, frame):
        frame = nn.ReLU()(self.conv1(frame))
        frame = nn.ReLU()(self.conv2(frame))
        frame = frame.reshape(-1, 40*14*14)
        frame = nn.ReLU()(self.dense(frame))
        mu = nn.Tanh()(self.mu(frame))
        var = nn.Softplus()(self.var(frame))
        value = self.critic(frame)
        return mu, var, value
        

class ActorCritic(nn.Module):
    def __init__(self, device):
        super(ActorCritic, self).__init__()
        self.model = Model().to(device)
        self.rewards = torch.Tensor([]).to(device)
        self.log_probabilities = torch.Tensor([]).to(device)
        self.action_values = torch.Tensor([]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.gamma = 0.99
        self.device = device
    
    def act(self, frame):
        mu, var, action_value = self.model(frame)
        action = Normal(loc=mu, scale=torch.sqrt(var)).sample()
        action[0][0] = torch.clamp(action[0][0], min=-1, max=1)
        action[0][1] = torch.clamp(action[0][1], min=0, max=1)
        action[0][2] = torch.clamp(action[0][2], min=0, max=1)
        
        log_prob = ActorCritic.calc_logprob(mu, var, action_value).to(device)
        
        return action, log_prob, action_value
        
    def add_to_trajectory(self, reward, log_probability, action_value):
        
        self.rewards = torch.cat((self.rewards, reward))
        self.log_probabilities = torch.cat((self.log_probabilities, log_probability))
        self.action_values = torch.cat((self.action_values, action_value))
    
    @staticmethod
    def calc_logprob(mu_v, var_v, actions_v):
        p1 = -((mu_v - actions_v) ** 2) / (2*var_v)
        p2 = -torch.log(torch.sqrt(2 * math.pi * var_v))
        return p1 + p2

    # loss = -log(N(a|mu, std)) * td_error
    # td_error = (discounted_return_vector - value(state))**2
    def update_model(self):
        self.optimizer.zero_grad()
        discounted_reward = 0
        for idx in range(self.rewards.shape[0] - 1, -1, -1):
            discounted_reward = self.rewards[idx] + self.gamma * discounted_reward
            self.rewards[idx] = discounted_reward
            
        self.rewards = self.rewards.unsqueeze(-1)
        td_error = self.rewards - self.action_values
        td_error = torch.mul(td_error, td_error)
        
        loss = -torch.mul(self.log_probabilities, td_error).mean()
        print("loss on episode {} is {}".format(ep_num, loss.item()))
        loss.backward()
        self.optimizer.step()
        
        self.rewards = torch.Tensor([]).to(device)
        self.log_probabilities = torch.Tensor([]).to(device)
        self.action_values = torch.Tensor([]).to(device)
        

# Figure out whether action value is supposed to be state value
if __name__ == "__main__":
    env = gym.make("CarRacing-v0")
    env = ProcessFrame96(env)
    device = "cuda" if torch.cuda.is_available else "cpu"
    
    agent = ActorCritic(device=device)
    optimizer = torch.optim.Adam(agent.parameters())
    
    num_episodes = 1000
    
    for ep_num in tqdm(range(num_episodes)):
        agent.model.zero_grad()
        state = env.reset()
        state = torch.Tensor(list(state)).permute(2, 0, 1).unsqueeze(0).to(device)
        total_reward = 0
        while True:
            if (ep_num + 1) % 10 == 0:
                env.render()
            
            action, log_prob, action_value = agent.act(state)
            state_next, reward, terminal, _ = env.step(action.detach().cpu().numpy()[0])
            
            total_reward += reward
            
            agent.add_to_trajectory(torch.Tensor([reward]).to(device), log_prob, action_value)

            state_next = torch.Tensor([state_next]).permute(0, 3, 1, 2).to(device)

            state = state_next
            if terminal:
                break
        print("The total reward on episode {} is {}".format(ep_num, total_reward))
        agent.update_model()
            
    
    
    env.close()
