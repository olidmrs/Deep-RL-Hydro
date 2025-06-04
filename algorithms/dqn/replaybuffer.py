import numpy as np
from collections import deque
import random
random.seed(1)
np.random.seed(1)

class ReplayBuffer():

    def __init__(self, size : int) -> None:
        """
        Initializes the queue of the replay buffer

        Args:
            size (int): size of the queue
        """        
        self.buffer = deque(maxlen = size)
    
    def add(self,
            s : float | int,
            a : float | int,
            r : float,
            next_s : float | int,
            done : int) -> deque:
        """
        Adds a transition to the queue

        Args:
            s (float | int): Intial state
            a (float | int): Action taken during transition
            r (float): Reward from transition
            next_s (float | int): Next state reached after transition
            done (int): If this transition was a terminating one

        Returns:
            (deque): Queue
        """        
        return self.buffer.append((s, a, r, next_s, done))

    def sample(self, batch_size : int):
        """
        Samples a batch from the replay buffer

        Args:
            batch_size (int): batch size to sample

        Returns:
            Returns a tuple of np.ndarrays for each components of step (s, a, r, next_s, done)
        """        
        s, a, r, next_s, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(s), np.array(a), np.array(r, dtype = np.float32), np.array(next_s), np.array(done, dtype=np.uint8)
    
    def __len__(self):
        return len(self.buffer)