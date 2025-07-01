from environment.hydroenv import HydroEnv
from environment.hydroenvt2 import HydroEnvt2

from algorithms import DynamicProgramming, Qlearning, Analytical
from algorithms.dqn import ReplayBuffer, DQNAgent
from algorithms.pgo import ReinforceAgentDiscrete
from algorithms.pgo import ReinforceAgentContinuous
from algorithms.pgo.networks import PolicyNetwork
from algorithms.pgo.networks import ValueNetwork
from algorithms.pgo import ActorCritic
from algorithms.pgo.ppo import PPO

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gym
from gym import spaces
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import random

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

episodes_reinforce = 300000
batch_size = 300

env = HydroEnv.Discret(
    t = 3,
    l_max = 100,
    l_min = 0,
    punition = -50
)

reinforce_discrete = ReinforceAgentDiscrete(
    input_dim = env.observation_space.shape[0],
    output_dim = env.action_space.n * 2,
    nb_hidden = 3,
    hidden_size = 64,
    gamma = 0.9,
    env = env,
    learning_rate = 0.0002, 
    learning_decay_rate = 1,
    final_learning_rate = 0.0001,
    beta = 0.1,
    decay_beta = 0.99999
)
discounted_rewards_batch = []
reward_history_reinforce_d = []
states_batch = []
actions_batch = []

for episode in range(episodes_reinforce):
    # print(episode)
    if episode % 1000 == 0: 
        print(f'Episode: {episode}, % of episodes: {episode/episodes_reinforce * 100} %')
        print(f'beta coefficient: {reinforce_discrete.beta:.10f}')
        
    
    actions, states, rewards = reinforce_discrete.gather_an_episode()
    discounted_rewards = reinforce_discrete.discount_rewards(rewards)
    discounted_rewards_batch.append(discounted_rewards)
    states_batch.append(states)
    actions_batch.append(actions)

    if len(discounted_rewards_batch) == batch_size:
        reinforce_discrete.update(states_batch, actions_batch, discounted_rewards_batch)
        discounted_rewards_batch = []
        states_batch = []
        actions_batch = []

    reward_history_reinforce_d.append(sum(rewards))
    reinforce_discrete.beta_decay()