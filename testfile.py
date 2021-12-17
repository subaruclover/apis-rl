import numpy as np
from agent import APIS, House

env = House()

observation = env.reset()
print(observation)
# [ 0.  0.  0. 50. 50.]
