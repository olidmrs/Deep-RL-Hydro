import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

from .networks.policynetwork import PolicyNetwork
from .networks.valuenetwork import ValueNetwork
from environment import HydroEnv


class PPO():
    def __init__(self,
                policynetwork : PolicyNetwork.Continuous,
                valuenetwork : ValueNetwork,
                gamma : float, 
                alpha_policynet : float,
                alpha_valuenet : float,
                env : HydroEnv.Continuous,
                alpha_decay : float,
                eps : float
                ):
        self.policynetwork = policynetwork
        self.valuenetwork = valuenetwork
        self.gamma = gamma
        self.alpha_policynet = alpha_policynet
        self.alpha_valuenet = alpha_valuenet
        self.env = env
        self.optimizer_policynet = optim.Adam(self.policynetwork.parameters(), lr = alpha_policynet)
        self.optimizer_valuenet = optim.Adam(self.valuenetwork.parameters(), lr = alpha_valuenet)
        self.criterion = nn.MSELoss()
        self.alpha_decay = alpha_decay
        self.eps = eps
        self.old_logits = torch.zeros(2)


    def update_policynet(self, states_batch, actions_batch, td_batch, logprob_batch) -> None:
        losses = []
        for state, action, advantage, last_logprob in zip(states_batch, actions_batch, td_batch, logprob_batch):
            output = self.policynetwork(state)
            mean = output[0]
            std = output[1]
            # print('__________')
            # print(f'mean: {mean}, std: {std}')

            dist = torch.distributions.Normal(mean, std)
            # old_dist = torch.distributions.Normal(last_logits[0], last_logits[1])
            # print(f'dist: {dist}')
            # print(f'old dist: {old_dist}')
            log_prob = dist.log_prob(action)
            # last_logprob = old_dist.log_prob(action)

            # print(f'log prob{log_prob}')
            # print(f'last log prob{last_logprob}')


            ratio = torch.exp(log_prob - last_logprob)
            clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            normalized_advantage = (advantage - np.mean(td_batch))/np.std(td_batch)

            # print(clipped_ratio, ratio)
            # print(normalized_advantage)
            loss = -torch.min(ratio * normalized_advantage, clipped_ratio * normalized_advantage)
            # print(loss)
            losses.append(loss)
        
        loss = torch.stack(losses).mean()
        self.optimizer_policynet.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policynetwork.parameters(), clip_value=1e-10)
        self.optimizer_policynet.step()
        
    def update_valuenet(self, actual_reward_batch, value_batch) -> None:
        actual_reward_batch = torch.FloatTensor(actual_reward_batch)
        value_batch = torch.stack(value_batch)
        loss = self.criterion(value_batch[:,0], actual_reward_batch)
        self.optimizer_valuenet.zero_grad()
        loss.backward()
        self.optimizer_valuenet.step()


    def td_error(self, reward : float, next_value : float, value : float, t : int) -> float:
        actual_reward = reward + self.gamma ** t * next_value
        return actual_reward, actual_reward - value
    
    def discount_rewards(self, rewards : list) -> list:
        discounted_returns = []
        for t in range(len(rewards)):
            g = 0
            for index, reward in enumerate(rewards[t:]):
                g += reward * (self.gamma ** index)
            discounted_returns.append(g)
        return discounted_returns 
    
    def sample_action(self, state : torch.Tensor):
        # print('___________')
        logit = self.policynetwork.forward(state)
        # print(type(logit))
        min_valid_action_space = max(0, state[0] + state[1] - self.env.l_max)
        max_valid_action_space = min(state[0] + state[1], self.env.l_max)
        # print(f'logit: {logit}')
        dist = torch.distributions.Normal(logit[0], logit[1])
        # print(f'dist{dist}')
        raw_action = dist.sample()
        # print(f'raw_action: {raw_action}')
        log_prob = dist.log_prob(raw_action)
        action = min_valid_action_space + torch.sigmoid(raw_action).item() * (max_valid_action_space - min_valid_action_space)
        return action, raw_action, log_prob

    def decay_alpha(self):
        self.alpha_policynet = self.alpha_policynet * self.alpha_decay