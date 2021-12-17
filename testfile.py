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

# importing random module
# import random
#
# random.seed(0)
#
# # print a random number between 1 and 1000.
# print(random.random())
#
# # if you want to get the same random number again then,
# random.seed(0)
# print(random.random())
#
# # If seed function is not used
#
# # Gives totally unpredictable responses.
# print(random.randint(1, 1000))

