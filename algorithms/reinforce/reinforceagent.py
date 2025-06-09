import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

from . import PolicyNetwork
from environment import HydroEnv

class ReinforceAgent():
    def __init__(
            self,
            input_dim : int,
            output_dim : int,
            nb_hidden : int,
            hidden_size : int,
            gamma : float, 
            learning_rate : float,
            env : HydroEnv,
            ) -> None:
        self.policynetwork = PolicyNetwork(input_dim, output_dim, nb_hidden, hidden_size)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policynetwork.parameters(), lr = learning_rate)
        self.criterion = nn.MSELoss()
        self.env = env
    
    def gather_an_episode(self) -> tuple[list,list,list]:
        self.env.reset()
        state = torch.tensor(self.env.state )
        done = False
        actions, states, rewards = [], [], []

        while not done:
            probs = self.policynetwork.forward(state)
            dist = torch.distributions.Categorical(probs = probs)
            action = dist.sample().item()
            next_state, reward, done, truncated, info = self.env.step(action)

            actions.append(torch.tensor(action, dtype=torch.int))
            states.append(state)
            rewards.append(reward)
            
            state = torch.tensor(next_state)
        return actions, states, rewards
    
    def discount_rewards(self, rewards : list) -> list:
        discounted_returns = []
        for t in range(len(rewards)):
            g = 0
            for index, reward in enumerate(rewards[t:]):
                g += reward * (self.gamma ** index)
            discounted_returns.append(g)
        return discounted_returns 
    
    def update(self, states, actions, discounted_returns):
        loss = 0
        for state, action, g in zip(states, actions, discounted_returns):
            probs = self.policynetwork.forward(state)
            dist = torch.distributions.Categorical(probs = probs)
            log_prob = dist.log_prob(action)

            loss += -log_prob * g

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        