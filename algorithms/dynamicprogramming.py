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

        v = np.zeros([self.env.t + 1, self.env.l_max + 1 + max(self.env.deterministic_inflows)])
        pi = np.zeros([self.env.t + 1, self.env.l_max + 1 + max(self.env.deterministic_inflows)])
        

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
                        reward, _ = self.env.get_current_reward(t, next_l, a)
                        Q_table.append(reward + v[t + 1, next_l])
                        actions.append(a)
                        # if self.env.l_min <= next_l  <= self.env.l_max:
                        #     Q_table.append(self.env.reward(t, l, a) + v[t + 1, next_l])
                        #     actions.append(a)
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

        for t in range(self.env.t):
            action = int(pi[t, l])
            optimal_pi.append(action)
            l = l + self.env.deterministic_inflows[t] - action
            waterlevel.append(l)
        return optimal_pi, waterlevel
    
    def extract_value(self, v : np.ndarray) -> float:
        """
        Extracts the value of starting at l0 and following pi

        Args:
            v (np.ndarray): Value table

        Returns:
            float: Value of starting at l0 and following pi
        """        
        return v[0, self.env.l_initial + self.env.waterinflows[0]]
    
