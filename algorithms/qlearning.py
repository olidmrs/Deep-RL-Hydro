import numpy as np
from environment.hydroenv import HydroEnv
import pickle
import os
import random

class Qlearning():

    def __init__(
            self,
            env : HydroEnv,
            gamma : float,
            episodes : int,
            learning_rate : float,
            learning_rate_decay: float,
            epsilon : float,
            epsilon_decay : float,
            min_epsilon : float = 0.1
            ) -> None:
        
        """
        Args:
            env (HydroEnv): Hydro system environment
            gamma (float): Discount parameter between 0 and 1
            episodes (int): Number of training episodes
            learning_rate (float): Learning rate by which the Q value is modified
            epsilon (float): Base epsilon to determine exploration vs exploitation of agent
            epsilon_decay (float): Decay rate between 0 and 1 of epsilon 
            min_epsilon (float): minimum epsilon to which we keep exploring 
        """        
        self.env = env
        self.gamma = gamma
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.episode_hit_min_eps = 0

        # Q_table is of dimensions [t, s, I, a]
        self.Q_table = np.zeros((self.env.t + 1, self.env.l_max + 1, self.env.l_max + 1, self.env.action_space.n * 2 + 1))
        self.visit_counts = np.zeros((self.env.t + 1, self.env.l_max + 1, self.env.l_max + 1, self.env.action_space.n * 2 + 1))
        self.epsilon_history = []
    
    def qlearning_solver(self) -> None:

        """
        Algorithm iterates through each episode by starting at a random state at time period 0. It then propagates
        through each period of our system taking an action at each period. The action is chosen with an 
        epsilon-greedy exploration vs exploitation method.
        """       

        count_exploration = 0
        count_exploitation = 0
        epsilon_history = []
        reward_history = []
        for ep in range(self.episodes):
            epsilon_history.append(self.epsilon)
            if ep % 10000 == 0:
                print(f'ep: {ep} %: {ep/self.episodes * 100} %')
                print(f'epsilon: {self.epsilon}')
                print(f'learning rate: {self.learning_rate} ')
        
                
            self.env.reset()
            episode_reward = 0
            for t in range(self.env.t):
                s = self.env.state[0]
                waterinflow = self.env.state[1]

                # Determination of action space with consideration of hydro system constraints
                possible_actions = [a for a in self.env.get_actions()]

                # Exploration
                if np.random.rand() <= self.epsilon:
                    count_exploration += 1
                    a = random.choice(possible_actions)
                    next_state, reward, done, truncated, info = self.env.step(a)
                
                # Exploitation
                else:
                    count_exploitation += 1
                    qvals = [self.Q_table[t, s, waterinflow, a] for a in possible_actions]
                    a = possible_actions[np.argmax(qvals)]
                    # a = possible_actions[np.argmax(self.Q_table[t, s, waterinflow, possible_actions])]
                    next_state, reward, done, truncated, info = self.env.step(a)
                
                episode_reward += reward

                # TD_error determination for non-final stage
                TD_error = reward + self.gamma * np.max(self.Q_table[t + 1, next_state[0], next_state[1], :]) - self.Q_table[t, s, waterinflow, a]
                
                # Q table update
                self.Q_table[t, s, waterinflow, a] = self.Q_table[t, s, waterinflow, a] + self.learning_rate * TD_error

                # Terminal Q value based on water level
                # if done:
                #     terminal_reward, _ = self.env.get_current_reward(t + 1, next_state[0], 0)
                #     self.Q_table[t + 1, next_state[0], next_state[1], a] = self.Q_table[t + 1, next_state[0], next_state[1], a] + self.learning_rate * (terminal_reward - self.Q_table[t + 1, next_state[0], next_state[1], a])
                #     episode_reward += terminal_reward
                
                self.visit_counts[t, s, waterinflow, a] = 1
            
            # Decay of epsilon for exploration
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.learning_rate = max(0.5, self.learning_rate * self.learning_rate_decay)
            reward_history.append(episode_reward)
            
        print(f'\n Exploration rate: {round(100 * count_exploration/(count_exploitation + count_exploration), 1)} %')
        print(f'Exploitation rate: {round(100 * count_exploitation/(count_exploitation + count_exploration), 1)} %')
        
        return epsilon_history, reward_history

    def extract_policy(self, initial_waterlevel : int, deterministic_inflows : list = []) -> tuple[list, list]:
        """
        Extracts optimal policy propagating through our system starting at initial state 

        Returns:
            pi (list): optimal actions to take at each time step starting at initial state
            waterlevel (list): waterlevel at each period when stating at initial state and following optimal policy
        """
        inflows = []
        l = initial_waterlevel
        pi = []
        waterlevel = [l]
        total_reward = 0
        for t in range(self.env.t + 1):

            if len(deterministic_inflows) == 0:
                if t != self.env.t:
                    inflows.append(self.env.get_inflow(t))
                    print(t)
                    print(l)
                    a = np.argmax(self.Q_table[t, l, self.env.inflow_cache[t], :])
                    pi.append(a)
                    reward, _ = self.env.get_current_reward(t, l, a)
                    l = l + self.env.inflow_cache[t] - a
                    total_reward += reward
                    waterlevel.append(l)
                else: 
                    reward, _ = self.env.get_current_reward(t,l,0)
                    total_reward += reward

            else:
                if t != self.env.t:
                    inflows = deterministic_inflows
                    a = np.argmax(self.Q_table[t, l, inflows[t], :])
                    pi.append(a)
                    reward, _ = self.env.get_current_reward(t, l, a)
                    l = l + inflows[t] - a
                    total_reward += reward
                    waterlevel.append(l)
                else:
                    reward, _ = self.env.get_current_reward(t,l,0)
                    total_reward += reward
        self.env.reset()
        return pi, waterlevel, inflows, total_reward
    
    def save_model(self, filename : str) -> None:
        """
        Saves object of class to models folder

        Args:
            filename (str): filename with .pkl extension 
        """        
        os.makedirs('models', exist_ok= True)
        filepath = os.path.join('models', filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(filepath: str) -> 'Qlearning':
        """
        Loads object of class to retrieve model

        Args:
            filepath (str): filepath of folder where model is saved

        Returns:
            model (Qlearning): object of class Qlearning
        """        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
        
    def extract_reward_of_pi(self, waterlevel : list, pi: list) -> float:
        reward = 0
        for t in range(self.env.t):
            reward += self.env.get_current_reward(t, waterlevel[t], pi[t])[0]
        return reward
    
    def compute_non_visited_count(self, waterlevel, waterinflow, pi) -> int:
        for t in range(self.env.t):
            if self.visit_counts[t, waterlevel[t], waterinflow[t], pi[t]] == 0.0:
                return 1
        return 0