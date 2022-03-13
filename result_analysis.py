"""
analyze the saved data (from ./saved/*.data)
Create by Qiong
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

"""
# e001
# load saved files
with open("saved/prio_memo_e001.data", "rb") as f1:
    data_e001 = pickle.load(f1)


action_req_low_e001 = [data_e001.tree.data[i][8] for i in range(24*55)]  # [6]~[8]
action_req_high_e001 = [data_e001.tree.data[i][6] for i in range(24*55)]
action_acc_e001 = [data_e001.tree.data[i][7] for i in range(24*55)]

plt.plot(action_req_high_e001[:100], 'g', label='action_req_high')
plt.show()

# dcdc_current
grid_current_e001 = [data_e001.tree.data[i][4] for i in range(24*55)]

# plt.plot(grid_current_e001, 'g', label='grid_current')
# plt.show()

reward_e001 = [data_e001.tree.data[i][9] for i in range(24*55)]  # reward(p2)

# e002
# load saved files
with open("saved/prio_memo_e002.data", "rb") as f2:
    data_e002 = pickle.load(f2)

action_req_low_e002 = [data_e002.tree.data[i][8] for i in range(24*55)]  # [6]~[8]
action_req_high_e002 = [data_e002.tree.data[i][6] for i in range(24*55)]
action_acc_e002 = [data_e002.tree.data[i][7] for i in range(24*55)]

# dcdc_current
grid_current_e002 =[data_e002.tree.data[i][4] for i in range(24*55)]

reward_e002 = [data_e002.tree.data[i][9] for i in range(24*55)]  # reward(p2)

# e003
# load saved files
with open("saved/prio_memo_e003.data", "rb") as f3:
    data_e003 = pickle.load(f3)

action_req_low_e003 = [data_e003.tree.data[i][8] for i in range(24*55)]  # [6]~[8]
action_req_high_e003 = [data_e003.tree.data[i][6] for i in range(24*55)]
action_acc_e003 = [data_e003.tree.data[i][7] for i in range(24*55)]

# dcdc_current
grid_current_e003 =[data_e003.tree.data[i][4] for i in range(24*55)]

reward_e003 = [data_e003.tree.data[i][9] for i in range(24*55)]  # reward(p2)

# e004
# load saved files
with open("saved/prio_memo_e004.data", "rb") as f4:
    data_e004 = pickle.load(f4)

action_req_low_e004 = [data_e004.tree.data[i][8] for i in range(24*55)]  # [6]~[8]
action_req_high_e004 = [data_e004.tree.data[i][6] for i in range(24*55)]
action_acc_e004 = [data_e004.tree.data[i][7] for i in range(24*55)]

# dcdc_current
grid_current_e004 = [data_e004.tree.data[i][4] for i in range(24*55)]

reward_e004 = [data_e004.tree.data[i][9] for i in range(24*55)]  # reward(p2)
"""


with open("saved/natural_memo_e001.data", "rb") as f1:
    data_e001 = pickle.load(f1)
#
reward_e001 = [data_e001[i][9] for i in range(24*30)]  # reward(p2)
ig_e001 = [data_e001[i][4] for i in range(24*30)]

plt.plot(reward_e001)
# plt.show()
#
with open("saved/natural_reward_e001.data", "rb") as f1:
    rew_e001 = pickle.load(f1)

plt.plot(rew_e001[:])
plt.show()