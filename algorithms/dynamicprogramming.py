import numpy as np
from environment.hydroenv import HydroEnv


class DynamicProgramming():
    def __init__(self, env: HydroEnv) -> None:
        self.env = env
    
    def dynamic_prog_solution2(self) -> tuple[np.ndarray, np.ndarray]:
        """ 
        Dynamically solves the problem with backward propagation

        Returns: 
            Value table (np.ndarray): table of value of starting in each state at each time
            Policy table (np.ndarray): table of best action to take in each state at each time
        """
        # max(self.env.deterministic_inflows)
        v = np.zeros([self.env.t + 1, self.env.l_max + 1])
        pi = np.zeros([self.env.t + 1, self.env.l_max + 1])
        

        for t in range(self.env.t, -1, -1):
            if t == self.env.t:
                for l in range(self.env.l_max + 1):
                    reward, _ = self.env.get_current_reward(t, l, 0)
                    v[t,l] = reward
            
            else:
                for l in range(self.env.l_max + 1):
                    Q_table = []
                    actions = []
                    for a in self.env.get_deterministic_actions(l, self.env.deterministic_inflows[t]):
                        next_l = l + self.env.deterministic_inflows[t] - a
                        reward, _ = self.env.get_current_reward(t, l, a)
                        Q_table.append(reward + v[t + 1, next_l])
                        actions.append(a)
                    if Q_table:
                        v[t,l] = np.max(Q_table)
                        pi[t,l] = actions[np.argmax(Q_table)]
        return v, pi

    def policy_extraction(self, pi : np.ndarray, waterlevel_t0: int) -> tuple[list, list]:

        """
        Extracts actions of optimal policy for each step considering starting waterlevel l_min

        Args: 
            pi (np.ndarray) : pi table indicating optimal action for each t/l pairs (state)
        
        Returns:
            optimal_pi (list): optimal actions for each step considering starting waterlevel
            waterlevel (list): waterlevel at each step
        """        
        l = waterlevel_t0
        optimal_pi = []
        waterlevel = [l]
        total_reward = 0
        for t in range(self.env.t + 1):
            if t != self.env.t:
                action = int(pi[t, l])
                optimal_pi.append(action)
                reward, _ = self.env.get_current_reward(t, l, action)
                l = l + self.env.deterministic_inflows[t] - action
                total_reward += reward
                waterlevel.append(l)
            else:
                reward, _ = self.env.get_current_reward(t,l,0)
                total_reward += reward
        return optimal_pi, waterlevel, total_reward

    
