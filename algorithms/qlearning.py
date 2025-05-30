import numpy as np
from environment.hydroenv import HydroEnv
import pickle
import os

class Qlearning():

    def __init__(
            self,
            env : HydroEnv,
            gamma : float,
            epoch : int,
            learning_rate : float,
            epsilon : float,
            epsilon_decay : float,
            min_epsilon : float = 0.1
            ) -> None:
        
        """
        Args:
            env (HydroEnv): Hydro system environment
            gamma (float): Discount parameter between 0 and 1
            epoch (int): Number of training episodes
            learning_rate (float): Learning rate by which the Q value is modified
            epsilon (float): Base epsilon to determine exploration vs exploitation of agent
            epsilon_decay (float): Decay rate between 0 and 1 of epsilon 
            min_epsilon (float): minimum epsilon to which we keep exploring 
        """        
        self.env = env
        self.gamma = gamma
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q_table = np.zeros((self.env.t, (self.env.l_max + 1) * 2, (self.env.l_max + 1) * 2))
    
    def qlearning_solver(self) -> None:

        """
        Algorithm iterates through each epoch by starting at a random state at time period 0. It then propagates
        through each period of our system taking an action at each period. The action is chosen with an 
        epsilon-greedy exploration vs exploitation method.
        """       

        count_exploration = 0

        for ep in range(self.epoch):
            if ep % 100000 == 0:
                print(f'ep: {ep}')

            initial_s = np.random.randint(self.env.l_min, self.env.l_max + 1)
            waterinflows = self.env.waterinflows

            for t in range(self.env.t):
                s = initial_s

                # Determination of action space with consideration of hydro system constraints
                possible_actions = [a for a in self.env.get_actions(s, waterinflows[t])]
                
                # Exploration
                if np.random.rand() <= self.epsilon:
                    count_exploration += 1
                    a = self.draw_random_action(possible_actions)
                    next_s = s + waterinflows[t] - a
                
                # Exploitation
                else:
                    a = possible_actions[np.argmax(self.Q_table[t, s, possible_actions])]
                    next_s = s + waterinflows[t] - a 
                
                # TD_error determination for final stage
                if t == self.env.t - 1:
                    TD_error = self.env.reward(t, s, a) 

                # TD_error determination for non-final stage
                else:
                    # print(f'min-max possible action: {min(possible_actions)} - {max(possible_actions)}')
                    # print(f'time: {t}, s: {s}, i: {waterinflows[t]}, a: {a}')
                    # print(f'next state: {next_s}')
                    # print(f'Q_table size: {self.Q_table.shape}')
                    TD_error = self.env.reward(t, s, a) + self.gamma * max(self.Q_table[t + 1, next_s, :]) - self.Q_table[t, s, a]
                
                # Q table update
                self.Q_table[t, s, a] = self.Q_table[t, s, a] + self.learning_rate * TD_error
                    
                initial_s = next_s
            
            # Decay of epsilon for exploration
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        print(f'Exploration rate: {round(100 * count_exploration/self.epoch, 1)} %')
    
    @staticmethod
    def draw_random_action(possible_actions : list) -> int:
        """
        Draws a random action from our action space

        Args:
            possible_actions (list): action space with consideration of hydro system constraints

        Returns:
            Action (int): action
        """        
        try:
            draw = np.random.randint(0, len(possible_actions))
            return possible_actions[draw]
        except:
            return 0

    def extract_policy(self) -> tuple[list, list]:
        """
        Extracts optimal policy propagating through our system starting at initial state 

        Returns:
            pi (list): optimal actions to take at each time step starting at initial state
            waterlevel (list): waterlevel at each period when stating at initial state and following optimal policy
        """        
        l = self.env.l_initial 
        pi = []
        waterlevel = [l]
        for t in range(self.env.t):
            a = np.argmax(self.Q_table[t, l, :])
            pi.append(a)
            l = l + self.env.waterinflows[t] - a
            waterlevel.append(l)
        return pi, waterlevel
    
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
        
    def extract_value_of_pi(self) -> float:
        """
        Extracts the value of starting at l0 and following pi

        Args:
            q_table (np.ndarray): Value table

        Returns:
            float: Value of starting at l0 and following pi
        """        
        return max(self.Q_table[0, self.env.l_initial, : ])
    