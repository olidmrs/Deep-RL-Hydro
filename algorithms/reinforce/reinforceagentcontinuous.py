import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

from .policynetwork import PolicyNetwork
from environment import HydroEnv



class ReinforceAgentContinuous():
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
        
        self.policynetwork = PolicyNetwork.Continuous(input_dim, output_dim, nb_hidden, hidden_size)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policynetwork.parameters(), lr = learning_rate)
        self.criterion = nn.MSELoss()
        self.env = env
    
    def gather_an_episode(self) -> tuple[list,list,list]:
        self.env.reset()
        state = torch.tensor(self.env.state, dtype=torch.float)
        done = False
        actions, states, rewards = [], [], []

        while not done:
            logit = self.policynetwork.forward(state)
            min_valid_action_space = max(0, self.env.state[0] + self.env.state[1] - self.env.l_max)
            max_valid_action_space = min(self.env.state[0] + self.env.state[1], self.env.l_max)
            dist = torch.distributions.Normal(logit[0], logit[1])
            raw_action = dist.sample()
            action = min_valid_action_space + torch.sigmoid(raw_action).item() * (max_valid_action_space - min_valid_action_space)

            next_state, reward, done, truncated, info = self.env.step(action)
            actions.append(torch.tensor(raw_action, dtype=torch.float))
            states.append(state)
            rewards.append(reward)
            state = torch.tensor(next_state, dtype=torch.float)
        return actions, states, rewards
    
    def discount_rewards(self, rewards : list) -> list:
        discounted_returns = []
        for t in range(len(rewards)):
            g = 0
            for index, reward in enumerate(rewards[t:]):
                g += reward * (self.gamma ** index)
            discounted_returns.append(g)
        return discounted_returns 
    
    def update(self, states_batch, actions_batch, discounted_rewards_batch):
        loss = 0
        all_returns = np.concatenate(discounted_rewards_batch)
        b = np.mean(all_returns)
        for states, actions, rewards in zip(states_batch, actions_batch, discounted_rewards_batch):
            for state, action, g in zip(states, actions, rewards):
                output = self.policynetwork.forward(state)
                dist = torch.distributions.Normal(output[0], output[1])
                log_prob = dist.log_prob(action)
                loss += -log_prob * (g - b)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()