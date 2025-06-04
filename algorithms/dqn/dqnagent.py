import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

from . import ReplayBuffer
from . import DQN
from environment import HydroEnv
import random

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

class DQNAgent():
    def __init__(
            self,
            input_dim : int,
            output_dim : int,
            nb_hidden : int,
            hidden_size : int,
            replay_buffer : ReplayBuffer,
            learning_rate : float,
            env : HydroEnv,
            gamma : float,
            init_eps : float,
            final_eps : float,
            eps_decay_rate : float
            ) -> None:
        """
        Initlizes the agent with quality network.

        Args:
            input_dim (int): Dimension of neural network input
            output_dim (int): Dimension of neural network output
            nb_hidden (int): Number of hidden layers
            hidden_size (int): Number of neurons per hidden layers
            replay_buffer (ReplayBuffer): Queue of past steps
            learning_rate (float): Learning rate used for the parameter optimizer
            env (HydroEnv): Environment where agent will interact
            gamma (float): Discount rate used for future actions when computing target q value
            init_eps (float): Initial epsilon used for action policy
            final_eps (float): Final (or minimal) epsilon used for action policy
            eps_decay_rate (float): Decay rate of the epsilon
        """        
        self.dqn = DQN(input_dim, output_dim, nb_hidden, hidden_size)
        self.target_dqn = DQN(input_dim, output_dim, nb_hidden, hidden_size)
        self.replay_buffer = replay_buffer

        # We pass self.dqn.parameters() to tell what parameters to update
        self.optimizer = optim.Adam(self.dqn.parameters(), lr = learning_rate)
        self.criterion = nn.MSELoss()
        self.env = env
        self.gamma = gamma
        self.init_eps = init_eps
        self.final_eps = final_eps
        self.eps = init_eps
        self.eps_decay_rate = eps_decay_rate

        self.exploration_episodes = 0
        self.exploitation_episodes = 0

    def choose_action(self, state : tuple[int, int, int]) -> int:
        """
        Chooses which action the agent will take following an epsilon-greedy policy by propagating forward 
        in the quality network.

        Args:
            state (tuple[int, int, int]): input which is the state/observation space (Lt, It, t)

        Returns:
            int: Action chosen
        """        
        min_valid_action_space = max(0, self.env.state[0] + self.env.state[1] - self.env.l_max)
        max_valid_action_space = min(self.env.state[0] + self.env.state[1], self.env.l_max) + 1
        # Exploitation
        if np.random.random() > self.eps:
            self.exploitation_episodes += 1
            state = torch.FloatTensor(state).unsqueeze(0)
            # Q_value is a tensor representing the value of each possible action 
            q_value = self.dqn.forward(state)
            q_value = q_value[0, min_valid_action_space : max_valid_action_space]
            # Action will be the index of the maximal q value
            # print(f'min: {min_valid_action_space}, max: {max_valid_action_space}')
            action = q_value.argmax().item()

        # Exploration
        else:
            self.exploration_episodes += 1
            action = random.randrange(max_valid_action_space)
        return action
    
    def update(self, batch_size : int) -> None:

        """
        Updates the parameters of the neural network. Samples a batch_size number of steps from the replay
        buffer. Computes temporal difference for our batch tensor. Uses temporal difference to optimize parameters
        to minimize loss function. This allows us to tweak the weights by propagating backwards through the neural
        network.

        Args:
            batch_size (int)): size of batch that we train
        """        
        if len(self.replay_buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)

        # Compute q values for all states in buffer sample
        q_values = self.dqn(state)
        # Compute q values for all next states in buffer sample
        next_q_values = self.target_dqn(next_state)

        
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) # Extract Q_values of action taken for each element of batch
        next_q_value = next_q_values.max(1)[0] # Get max value for the next state
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = self.criterion(q_value, expected_q_value.detach())
        self.optimizer.zero_grad() # Resets gradient
        loss.backward() # Computes gradients of loss with respect to all parameters
        self.optimizer.step() # Perform optimization

    def epsilon_decay(self) -> None:
        """
        Computes the decay of epsilon.
        """        
        self.eps = max(self.eps * self.eps_decay_rate, self.final_eps)

    def update_target_network(self):
        """
        Updates target network from neural network parameters
        """        
        self.target_dqn.load_state_dict(self.dqn.state_dict())