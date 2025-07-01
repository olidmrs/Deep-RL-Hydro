import numpy as np
from environment import HydroEnvt2

class Analytical():
    def __init__(self):
        pass

    def solve_for_t(env : HydroEnvt2.Discret, t : int, l, i):
        if t == 1:
            return 1/3 * (l + i - 1)
        else:
            gamma = 1 + l + i
            beta = 2 + l + 4 * i
            a = 5
            b = 4 - 2 * beta - 4 * gamma
            c = gamma * beta - beta - 3 * gamma

            x1, x2 = Analytical.quadratic(a, b, c)
            

            valid = []
            for x in [x1, x2]:
                if x < 0 or x > l + i:
                    continue
                L1 = l + i - x
                if not (env.l_min <= L1 <= env.l_max):
                    continue
                r, _ = env.get_current_reward(t, l, x)
                valid.append((x,r))

            return max(valid, key= lambda tup: tup[1])[0]
    
    def quadratic(a, b, c):
        x1 = (-b - np.sqrt(b ** 2 - 4 * a * c))/(2 * a)
        x2 = (-b + np.sqrt(b ** 2 - 4 * a * c))/(2 * a)
        return x1, x2
    
    

