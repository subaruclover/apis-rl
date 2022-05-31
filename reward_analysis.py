"""
analyze reward data from
1) saved data (from ./saved/*.data)

compare p2, reward in different settings (with default value)

Create by Qiong
"""

import os
import pandas as pd
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

sns.set(style="whitegrid")

# e001~e004
# load saved files
with open("saved/prio_reward_e001_May_iter5_time.data", "rb") as f1:
    prio_reward_e001 = pickle.load(f1)

with open("saved/prio_reward_e002_May_iter5_time.data", "rb") as f2:
    prio_reward_e002 = pickle.load(f2)

with open("saved/prio_reward_e003_May_iter5_time.data", "rb") as f3:
    prio_reward_e003 = pickle.load(f3)

with open("saved/prio_reward_e004_May_iter5_time.data", "rb") as f4:
    prio_reward_e004 = pickle.load(f4)

plt.plot(prio_reward_e001, 'g', label='reward e001')
# plt.plot(prio_reward_e002, 'b', label='reward e002')
# plt.plot(prio_reward_e003, 'r', label='reward e003')
# plt.plot(prio_reward_e004, 'y', label='reward e004')

plt.legend()
plt.show()