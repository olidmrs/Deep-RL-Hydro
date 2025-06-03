import gym.spaces
import numpy as np
import gym

class HydroEnv(gym.Env):
    """
    Observation Space:
    Represents the state at which the agent is now at. It is the information
    required by the agent to take a decision. It is the input layer of our Neural Network.
    Observation space = [Lt, It, t]
    Where Lt is the water level at current period.
    Where It is the inflow of water incoming during period t.
    Where t is the period at which the agent is currently at

    Action Space:
    Represents all the set of possible actions.

    Reward:
    Reward is computed based on the amount of turbined water and water level for non terminal periods. Terminal
    periods have a reward based only on the water level. If the constraints of the system are broken, a penality
    is associated leading to truncation of the episode.

    Termination:
    Termination is reaches all periods t are done or once an episode is truncated from constraint violation.
    """    

    def __init__(
            self,
            t : int,
            l_max : int,
            l_min : int,
            punition : int,
            ) -> None:
        """
        Initialize observation space and action space

        Args:
            t (int): number of periods where terminal period requires no decison/action
            l_max (int): maximum reservoir water level
            l_min (int): minimum reservoir water level
            state (int): state at which we start and initialize the system
            punition (int) : punition for constraint breaking
        """
        self.t = t
        self.l_max = l_max
        self.l_min = l_min
        self.punition = punition
        self.observation_space = gym.spaces.MultiDiscrete(np.array([l_max, l_max, t]))
        self.action_space = gym.spaces.Discrete(l_max + 1)
        self.state = 0
        self.last_i = 0
        self.current_t = 0
        self.inflow_cache = []

    def step(self, action : int) -> tuple[int, float, bool, bool, dict]:
        """
        Computes a step or transition of our state following a decision from the agent

        Args:
            action (int): Action taken from the agent

        Returns:
            tuple[int, float, bool, bool, dict]: Returns a tuple of the next state, the reward associated with 
            the transition, a bool of if the step led to termination, a bool of if the step led to truncation,
            and an information dictionnary (not implemented here).
        """        
        inflow = self.state[1]
        next_waterlevel = self.state[0] + inflow - action
        reward, truncated = self.get_current_reward(self.current_t, next_waterlevel, action)

        # Updates
        self.last_i = inflow
        done = 0
        
        if self.current_t == self.t and truncated != 1:
            done = 1

        self.current_t += 1

        self.state = (next_waterlevel, self.get_inflow(self.current_t), self.current_t)
        return self.state, reward, done, truncated, {}

    def reset(self) -> None:
        """
        Resets the environment for it to be ready for a new epsideo
        """        
        self.current_t = 0
        self.inflow_cache = []
        self.state = (np.random.randint(0, self.l_max), self.get_inflow(self.current_t), self.current_t)

    def get_current_reward(self, t : int, next_waterlevel : int, action : int) -> tuple[float, int]:
        """
        Computes the current reward receives with a transition

        Args:
            t (int): Period
            next_waterlevel (int): The next water level following the transition
            action (int): The amount of water that was turbined

        Returns:
            tuple[float, int]: Returns a tuple of the reward and truncation bool
        """        
        if  next_waterlevel < self.l_min or next_waterlevel > self.l_max:
            return self.punition, 1
        else:    
            if t == self.t:
                return np.log((1 + next_waterlevel) ** 2), 0
            else:
                return np.log((1 + next_waterlevel) * (1 + action)), 0

    def get_inflow(self, t : int) -> int:
        """
        Computes the inflow for a given period and previous inflow of water.

        Args:
            t (int): period

        Returns:
            int: returns inflow of water coming between t and t + 1 clipped between reservoir constraints
        """        
        match t:
            case 0:
                i = max(10 + 5 * np.random.normal(loc = 0, scale = 1), 0)
            case 1:
                i = max(self.inflow_cache[t-1] + 5 * np.random.normal(loc = 0, scale = 1), 0)
            case 2:
                i = max(2 * self.inflow_cache[t-1] + 5 * np.random.normal(loc = 0, scale = 1), 0)
            case _:
                return 0
        self.inflow_cache.append(np.clip(int(i), self.l_min, self.l_max))
        return np.clip(int(i), self.l_min, self.l_max)
    
    def get_actions(self):
        waterlevel = self.state[0]
        inflow = self.state[1]

        a_min = max(waterlevel + inflow - self.l_max, 0)
        a_max = waterlevel + inflow - self.l_min
        return range(a_min, a_max + 1)