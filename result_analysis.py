"""
analyze the saved data (from ./saved/*.data)
Create by Qiong
"""

import os
import pandas as pd
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

"""
with open("saved/natural_memo_e001.data", "rb") as f1:
    data_e001 = pickle.load(f1)
#
reward_e001 = [data_e001[i][9] for i in range(24*30)]  # reward(p2)
ig_e001 = [data_e001[i][4] for i in range(24*30)]

# plt.plot(reward_e001)
# plt.show()
#
with open("saved/natural_reward_e001.data", "rb") as f1:
    rew_e001 = pickle.load(f1)

with open("saved/natural_reward_e002.data", "rb") as f1:
    rew_e002 = pickle.load(f1)

with open("saved/natural_reward_e003.data", "rb") as f1:
    rew_e003 = pickle.load(f1)

with open("saved/natural_reward_e004.data", "rb") as f1:
    rew_e004 = pickle.load(f1)

plt.plot(rew_e001[:], 'g-', label='reward_e001')
plt.plot(rew_e002[:], 'b*', label='reward_e002')
plt.plot(rew_e003[:], 'k--', label='reward_e003')
plt.plot(rew_e004[:], 'r-.', label='reward_e004')

plt.xlabel("every 3 hours")
plt.legend()
plt.show()
"""

"""
with open("saved/natural_memo_e001.data", "rb") as f1:
    data_e001 = pickle.load(f1)
#
reward_e001 = [data_e001[i][9] for i in range(24*30)]  # reward(p2)
ig_e001 = [data_e001[i][4] for i in range(24*30)]

# plt.plot(reward_e001)
# plt.show()
#
with open("saved/natural_reward_e001_May_iter3.data", "rb") as f1:
    rew_e001 = pickle.load(f1)

with open("saved/natural_reward_e002_May_iter3.data", "rb") as f1:
    rew_e002 = pickle.load(f1)

with open("saved/natural_reward_e003.data", "rb") as f1:
    rew_e003 = pickle.load(f1)

with open("saved/natural_reward_e004_May_iter3.data", "rb") as f1:
    rew_e004 = pickle.load(f1)

plt.plot(rew_e001[:], 'g-', label='reward_e001')
# plt.plot(rew_e002[:], 'b*', label='reward_e002')
# plt.plot(rew_e003[:], 'k--', label='reward_e003')
# plt.plot(rew_e004[:], 'r-.', label='reward_e004')

plt.xlabel("every 3 hours")
plt.legend()
plt.show()
"""

# """
output_sum_May_default = "oist_summary_May_default.csv"
output_sum_May_default_2 = "oist_summary_May_default_2.csv"
oist_summary_May_defa_lin = "oist_summary_May_defa_lin.csv"

output_sum_May_iter1 = "oist_summary_May_iter1_3.csv"
output_sum_May_iter1_shuf = "oist_summary_May_iter1_shuffle.csv"
output_sum_May_iter3 = "oist_summary_May_iter3.csv"
output_sum_May_iter1_1hr = "oist_summary_May_iter1_1hr.csv"

output_getpath = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
output_dir = output_getpath + "/apis-emulator/data/output"

output_file_default = os.path.join(output_dir, output_sum_May_default)
# output_file_default_2 = os.path.join(output_dir, output_sum_May_default_2)
output_file_default_lin = os.path.join(output_dir, oist_summary_May_defa_lin)
output_file_iter1 = os.path.join(output_dir, output_sum_May_iter1)
output_file_iter1_shuf = os.path.join(output_dir, output_sum_May_iter1_shuf)
output_file_iter3 = os.path.join(output_dir, output_sum_May_iter3)
output_file_iter1_1hr = os.path.join(output_dir, output_sum_May_iter1_1hr)

data_default = pd.read_csv(output_file_default)
# data_default_2 = pd.read_csv(output_file_default_2)
data_default_lin = pd.read_csv(output_file_default_lin)
data_iter1 = pd.read_csv(output_file_iter1)
data_iter1_shuf = pd.read_csv(output_file_iter1_shuf)
data_iter3 = pd.read_csv(output_file_iter3)
data_iter1_1hr = pd.read_csv(output_file_iter1_1hr)

dcdc_default = data_default['wg'][0:30]
# dcdc_default_2 = data_default_2['wg'][0:30]
dcdc_default_lin = data_default_lin['wg'][0:30]
dcdc_iter1 = data_iter1['wg'][0:30]
dcdc_iter1_shuf = data_iter1_shuf['wg'][0:30]
dcdc_iter3 = data_iter3['wg'][0:90]
dcdc_iter1_1hr = data_iter1_1hr['wg'][0:30]

acin_default = data_default['acin']
acin_iter1 = data_iter1['acin']

wasted_default = data_default['wasted']
wasted_iter1 = data_iter1['wasted']

ssr_pv_default = data_default['ssr_pv']
ssr_pv_iter1 = data_iter1['ssr_pv']

# bar plot of sum
acin_default_sum = acin_default[31]
acin_iter1_sum = acin_iter1[31]
wasted_default_sum = wasted_default[31]
wasted_iter1_sum = wasted_iter1[31]

data = [[acin_default_sum, wasted_default_sum],
[acin_iter1_sum, wasted_iter1_sum]]
X = np.arange(2)

# fig, ax = plt.subplots(1, 1)
# # ax2 = ax.twinx()
# # ax = fig.add_axes([0, 0, 1, 1])
# ax.bar(X + 0.00, data[0], width=0.25)
# ax.bar(X + 0.25, data[1], width=0.25)
# ax.legend(labels=['sum of default', 'sum of DQN'])
# # ax.set_ylabel('Power [W]')
# ax.set_title('purchased and wasted power [W]')
# plt.xticks([0.1, 1.1], ['purchased', 'wasted'])

# plt.plot(acin_default[0:30], 'm--', label='default purchased power')
# plt.plot(acin_iter1[0:30], 'g*-', label='DQN purchased power')

# plt.plot(ssr_pv_default[0:30], 'r--+', label='default ssr')
# plt.plot(ssr_pv_iter1[0:30], 'g*-', label='DQN ssr')

# plt.plot(dcdc_default, 'm--', label='default exchanged power')
# plt.plot(dcdc_default_2, 'c--', label='default exchanged power macbook')
# plt.plot(dcdc_default_lin, 'm--', label='default exchanged power')
#
plt.plot(dcdc_iter1, 'go-', label='DQN exchanged power, iter=1, 3hrs')
plt.plot(dcdc_iter1_shuf, 'b*-', label='DQN exchanged power, shuffle, iter=1, 3hrs')
# plt.plot(dcdc_iter3, 'go-', label='DQN exchanged power, iter=3')
# plt.plot(dcdc_iter1_1hr, 'k--', label='DQN exchanged power, iter=1, 1hr')

# #
# plt.xlabel("Days")
# plt.ylabel("Power [W]")
# # plt.ylabel("Rate")
# # plt.ylim(0, 1)
plt.legend(loc='upper right')
plt.show()
# """