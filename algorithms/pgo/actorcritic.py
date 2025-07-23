from environment import HydroEnv
from algorithms.pgo.networks.policynetwork import PolicyNetwork
from algorithms.pgo.networks.valuenetwork import ValueNetwork
import torch.optim as optim
import torch.nn as nn
import torch 
import numpy as np

class ActorCritic():
    def __init__(
            self,
            policynetwork : PolicyNetwork.Continuous,
            valuenetwork : ValueNetwork,
            gamma : float, 
            alpha_policynet : float,
            alpha_valuenet : float,
            env : HydroEnv.Continuous,
            alpha_decay : float
            ) -> None:
        
        self.policynetwork = policynetwork
        self.valuenetwork = valuenetwork
        self.gamma = gamma
        self.alpha_policynet = alpha_policynet
        self.alpha_valuenet = alpha_valuenet
        self.alpha_decay = alpha_decay
        self.optimizer_policynet = optim.Adam(self.policynetwork.parameters(), lr = alpha_policynet)
        self.optimizer_valuenet = optim.Adam(self.valuenetwork.parameters(), lr = alpha_valuenet)
        self.criterion = nn.MSELoss()
        self.env = env

    def td_error(self, reward : float, next_value : float, value : float, t : int) -> float:
        actual_reward = reward + self.gamma ** t * next_value
        # print(actual_reward, reward, value)
        return actual_reward, actual_reward - value

    def update_policynet(self, states_batch, actions_batch, td_batch, period_batch) -> None:
        losses = []
        for state, action, value, t in zip(states_batch, actions_batch, td_batch, period_batch):
            output = self.policynetwork(state)
            mean = output[0]
            std = output[1]
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action)
            # print(f'value {value}, td mean {np.mean(td_batch)}, td std {np.std(td_batch)}')
            # normalized_value = (value - np.mean(td_batch))/np.std(td_batch)

            # print(f'Normalized value: {normalized_value}')
            losses.append(-log_prob * value * self.gamma ** t)
        
        loss = torch.stack(losses).mean()
        # print(f'average loss {loss}')
        self.optimizer_policynet.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policynetwork.parameters(), clip_value=1e-11)
        self.optimizer_policynet.step()
        
    def update_valuenet(self, actual_reward_batch, value_batch) -> None:
        actual_reward_batch = torch.FloatTensor(actual_reward_batch)
        value_batch = torch.stack(value_batch)
        loss = self.criterion(value_batch[:,0], actual_reward_batch)
        self.optimizer_valuenet.zero_grad()
        loss.backward()
        self.optimizer_valuenet.step()
    
    def sample_action(self, state : torch.Tensor):
        logit = self.policynetwork.forward(state)
        min_valid_action_space = max(0, state[0] + state[1] - self.env.l_max)
        max_valid_action_space = min(state[0] + state[1], self.env.l_max)
        

        dist = torch.distributions.Normal(logit[0], logit[1])
        raw_action = dist.sample()
        # print(f'State: {state[0]}, Inflow: {state[1]}')
        # print(f'min et max: {min_valid_action_space, max_valid_action_space}')
        # print(f'action between 0 and 1 {torch.sigmoid(raw_action).item()}')
        action = min_valid_action_space + torch.sigmoid(raw_action).item() * (max_valid_action_space - min_valid_action_space)
        return action, raw_action

    def decay_alpha(self):
        self.alpha_policynet = self.alpha_policynet * self.alpha_decay