import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

from .policynetwork import PolicyNetwork
from environment import HydroEnv

class ReinforceAgentDiscrete():
    def __init__(
            self,
            input_dim : int,
            output_dim : int,
            nb_hidden : int,
            hidden_size : int,
            gamma : float, 
            learning_rate : float,
            env : HydroEnv,
            learning_decay_rate : float,
            final_learning_rate : float,
            beta : float,
            beta_decay_rate : float
            ) -> None:
        
        self.policynetwork = PolicyNetwork.Discret(input_dim, output_dim, nb_hidden, hidden_size)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policynetwork.parameters(), lr = learning_rate)
        self.criterion = nn.MSELoss()
        self.env = env
        self.learning_rate_decay = learning_decay_rate
        self.final_learning_rate = final_learning_rate
        self.beta = beta
        self.beta_decay_rate = beta_decay_rate
    
    def gather_an_episode(self) -> tuple[list,list,list]:
        self.env.reset()
        state = torch.tensor(self.env.state, dtype=torch.float)
        done = False
        actions, states, rewards = [], [], []

        while not done:
            logits = self.policynetwork.forward(state)
            min_valid_action_space = max(0, self.env.state[0] + self.env.state[1] - self.env.l_max)
            max_valid_action_space = min(self.env.state[0] + self.env.state[1], self.env.l_max) + 1
            mask = torch.zeros_like(logits)
            mask[min_valid_action_space : max_valid_action_space] = 1
            logits[mask == 0] = -1e9

            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().item()   

            next_state, reward, done, truncated, info = self.env.step(action)
            actions.append(torch.tensor(action, dtype=torch.int))
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
                logits = self.policynetwork.forward(state)
                min_valid_action_space = max(0, state[0].item() + state[1].item() - self.env.l_max)
                max_valid_action_space = min(state[0].item() + state[1].item(), self.env.l_max) + 1
                mask = torch.zeros_like(logits)
                mask[int(min_valid_action_space) : int(max_valid_action_space)] = 1

                logits[mask == 0] = -1e9
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs = probs)
                log_prob = dist.log_prob(action)
                loss += -log_prob * (g - b) 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def alpha_decay(self) -> None:
        """
        Computes the decay of epsilon.
        """
        self.learning_rate = max(self.learning_rate * self.learning_rate_decay, self.final_learning_rate)

    def beta_decay(self) -> None:
        """
        Computes the decay of epsilon.
        """
        self.beta = self.beta * self.beta_decay_rate