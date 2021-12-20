import numpy as np
from agent import APIS, House

env = House()
env.seed(0)
print(env.seed(30))

observation = env.reset()
print(observation)
# [ 0.  0.  0. 50. 50.]

# random module is imported
import random



