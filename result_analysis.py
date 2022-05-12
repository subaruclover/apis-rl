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
from matplotlib.pyplot import figure

sns.set(style="whitegrid")

"""
# e001
# load saved files
with open("saved/natural_memo_e001_May_iter1_3.data", "rb") as f1:
    data_e001 = pickle.load(f1)


# action_req_low_e001 = [data_e001.tree.data[i][8] for i in range(24*55)]  # [6]~[8], prior_data
action_req_low_e001 = [data_e001.tree.data[i][8] for i in range(24*55)]  # [6]~[8], natural data
action_req_high_e001 = [data_e001.tree.data[i][6] for i in range(24*55)]
action_acc_e001 = [data_e001.tree.data[i][7] for i in range(24*55)]

# dcdc_current
grid_current_e001 = [data_e001.tree.data[i][4] for i in range(24*55)]

# plt.plot(grid_current_e001, 'g', label='grid_current')
# plt.show()

reward_e001 = [data_e001.tree.data[i][9] for i in range(24*55)]  # reward(p2)

# e002
# load saved files
with open("saved/natural_memo_e002_May_iter1_3.data", "rb") as f2:
    data_e002 = pickle.load(f2)

action_req_low_e002 = [data_e002.tree.data[i][8] for i in range(24*55)]  # [6]~[8]
action_req_high_e002 = [data_e002.tree.data[i][6] for i in range(24*55)]
action_acc_e002 = [data_e002.tree.data[i][7] for i in range(24*55)]

# dcdc_current
grid_current_e002 =[data_e002.tree.data[i][4] for i in range(24*55)]

reward_e002 = [data_e002.tree.data[i][9] for i in range(24*55)]  # reward(p2)

# e003
# load saved files
with open("saved/natural_memo_e003.data_May_iter1_3", "rb") as f3:
    data_e003 = pickle.load(f3)

action_req_low_e003 = [data_e003.tree.data[i][8] for i in range(24*55)]  # [6]~[8]
action_req_high_e003 = [data_e003.tree.data[i][6] for i in range(24*55)]
action_acc_e003 = [data_e003.tree.data[i][7] for i in range(24*55)]

# dcdc_current
grid_current_e003 =[data_e003.tree.data[i][4] for i in range(24*55)]

reward_e003 = [data_e003.tree.data[i][9] for i in range(24*55)]  # reward(p2)

# e004
# load saved files
with open("saved/natural_memo_e004_May_iter1_3.data", "rb") as f4:
    data_e004 = pickle.load(f4)

action_req_low_e004 = [data_e004.tree.data[i][8] for i in range(24*55)]  # [6]~[8]
action_req_high_e004 = [data_e004.tree.data[i][6] for i in range(24*55)]
action_acc_e004 = [data_e004.tree.data[i][7] for i in range(24*55)]

# dcdc_current
grid_current_e004 = [data_e004.tree.data[i][4] for i in range(24*55)]

reward_e004 = [data_e004.tree.data[i][9] for i in range(24*55)]  # reward(p2)

figure(figsize=(25, 5), dpi=180)
plt.plot(action_req_high_e001[:100], 'g', label='action_req_high')


plt.title('E001~E004, action_req, iter=1')
plt.legend(loc='upper right')
plt.gca().set_aspect('auto')
plt.show()
plt.show()

"""

"""
# e001
# load saved files
with open("saved/natural_memo_e001_May_iter1_3.data", "rb") as f1:
    data_e001 = pickle.load(f1)


# action_req_low_e001 = [data_e001.tree.data[i][8] for i in range(24*55)]  # [6]~[8], prior_data
action_req_low_e001 = [data_e001[i][10] for i in range(24*55)]  # [6]~[8], natural data
action_req_high_e001 = [data_e001[i][8] for i in range(24*55)]
action_acc_e001 = [data_e001[i][9] for i in range(24*55)]

# dcdc_current
grid_current_e001 = [data_e001[i][4] for i in range(24*55)]

# plt.plot(grid_current_e001, 'g', label='grid_current')
# plt.show()

reward_e001 = [data_e001[i][11] for i in range(24*55)]  # reward(p2)

# e002
# load saved files
with open("saved/natural_memo_e002_May_iter1_time.data", "rb") as f2:
    data_e002 = pickle.load(f2)

action_req_low_e002 = [data_e002[i][10] for i in range(24*55)]  # [6]~[8]
action_req_high_e002 = [data_e002[i][8] for i in range(24*55)]
action_acc_e002 = [data_e002[i][9] for i in range(24*55)]

# dcdc_current
grid_current_e002 =[data_e002[i][4] for i in range(24*55)]

reward_e002 = [data_e002[i][11] for i in range(24*55)]  # reward(p2)

# e003
# load saved files
with open("saved/natural_memo_e003_May_iter1_time.data", "rb") as f3:
    data_e003 = pickle.load(f3)

action_req_low_e003 = [data_e003[i][10] for i in range(24*55)]  # [6]~[8]
action_req_high_e003 = [data_e003[i][8] for i in range(24*55)]
action_acc_e003 = [data_e003[i][9] for i in range(24*55)]

# dcdc_current
grid_current_e003 =[data_e003[i][4] for i in range(24*55)]

reward_e003 = [data_e003[i][11] for i in range(24*55)]  # reward(p2)

# e004
# load saved files
with open("saved/natural_memo_e004_May_iter1_time.data", "rb") as f4:
    data_e004 = pickle.load(f4)

action_req_low_e004 = [data_e004[i][10] for i in range(24*55)]  # [6]~[8]
action_req_high_e004 = [data_e004[i][8] for i in range(24*55)]
action_acc_e004 = [data_e004[i][9] for i in range(24*55)]

# dcdc_current
grid_current_e004 = [data_e004[i][4] for i in range(24*55)]

reward_e004 = [data_e004[i][11] for i in range(24*55)]  # reward(p2)

# action analyze
actions_space = np.linspace(0.2, 0.9, 8).tolist()


def get_int_action_values(action, action_space):
    actions, actions_value = [], []
    for i in range(240):  # 8(each day) * 30(days)
        action_req_hi = int(action[i])
        actions.append(action_req_hi)

    for i in range(240):
        action_value = action_space[actions[i]]
        actions_value.append(action_value)

    # transfer actions_value to 100% unit
    actions_value = [i * 100 for i in actions_value]

    return actions, actions_value


# get actions_value (3 sets for each house)
actions_req_hi_e001, act_req_high_value_e001 = \
    get_int_action_values(action_req_high_e001, actions_space)
actions_req_hi_e002, act_req_high_value_e002 = \
    get_int_action_values(action_req_high_e002, actions_space)
actions_req_hi_e003, act_req_high_value_e003 = \
    get_int_action_values(action_req_high_e003, actions_space)
actions_req_hi_e004, act_req_high_value_e004 = \
    get_int_action_values(action_req_high_e004, actions_space)

actions_req_low_e001, act_req_low_value_e001 = \
    get_int_action_values(action_req_low_e001, actions_space)
actions_req_low_e002, act_req_low_value_e002 = \
    get_int_action_values(action_req_low_e002, actions_space)
actions_req_low_e003, act_req_low_value_e003 = \
    get_int_action_values(action_req_low_e003, actions_space)
actions_req_low_e004, act_req_low_value_e004 = \
    get_int_action_values(action_req_low_e004, actions_space)

actions_acc_e001, act_acc_value_e001 = \
    get_int_action_values(action_acc_e001, actions_space)
actions_acc_e002, act_acc_value_e002 = \
    get_int_action_values(action_acc_e002, actions_space)
actions_acc_e003, act_acc_value_e003 = \
    get_int_action_values(action_acc_e003, actions_space)
actions_acc_e004, act_acc_value_e004 = \
    get_int_action_values(action_acc_e004, actions_space)

figure(figsize=(25, 5), dpi=180)
days = 30  # 7
N = 8 * days  # 8(each day) * 7(days)
x_axis = np.arange(0, N, 1)
# E001
plt.plot(act_req_high_value_e001[-N:], 'b')  # , label='action_req_high, E001')
plt.plot(act_req_low_value_e001[-N:], 'y')  # , label='action_req_low, E001')
plt.plot(act_acc_value_e001[-N:], 'g')  # , label='action_acc, E001')

plt.fill_between(x_axis, 100 + 0 * x_axis, act_acc_value_e001[-N:], alpha=.3, ec="w", hatch='/', label='Accept Discharge')
plt.fill_between(x_axis, 100 + 0 * x_axis, act_req_high_value_e001[-N:],  color='b', alpha=.5, label='Request Discharge')
plt.fill_between(x_axis, 0 * x_axis, act_acc_value_e001[-N:], alpha=.5, ec="w", hatch='\\', label='Accept Charge')
plt.fill_between(x_axis, 0 * x_axis, act_req_low_value_e001[-N:], color='g', alpha=.5, label='Request Charge')

# plt.fill_between(x_axis, act_req_high_value_e001[-N:], act_req_low_value_e001[-N:], alpha=.4)

# E003
# plt.plot(act_req_high_value_e003[-N:], 'b')  # , label='action_req_high, E001')
# plt.plot(act_req_low_value_e003[-N:], 'y')  # , label='action_req_low, E001')
# plt.plot(act_acc_value_e003[-N:], 'g')  # , label='action_acc, E001')

# plt.fill_between(x_axis, 100 + 0 * x_axis, act_acc_value_e003[-N:], alpha=.3, ec="w", hatch='/', label='Accept Discharge')
# plt.fill_between(x_axis, 100 + 0 * x_axis, act_req_high_value_e003[-N:],  color='b', alpha=.5, label='Request Discharge')
# plt.fill_between(x_axis, 0 * x_axis, act_acc_value_e003[-N:], alpha=.5, ec="w", hatch='\\', label='Accept Charge')
# plt.fill_between(x_axis, 0 * x_axis, act_req_low_value_e003[-N:], color='g', alpha=.5, label='Request Charge')

# plt.fill_between(x_axis, act_req_high_value_e003[-N:], act_req_low_value_e003[-N:], color='g', alpha=.4)
# plt.fill_between(x_axis, 0 * x_axis, act_req_low_value_e003[-N:], color='g', alpha=.5, label='Request Charge')

# plt.title('{} days of action values, E001, iter=1'.format(days))
plt.title('Last week ({} days) of action values, E001, iter=1'.format(days))
plt.xlabel("Every 3 hours")
plt.ylabel("RSOC [%]")
plt.xlim(0, N)
plt.ylim(0, 100)
plt.yticks(np.arange(0, 110, 10))
plt.legend(loc='upper right')
plt.gca().set_aspect('auto')
plt.show()

"""

"""
with open("saved/natural_memo_e001_May_iter3.data", "rb") as f1:
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

with open("saved/natural_reward_e003.data_May_iter3", "rb") as f1:
    rew_e003 = pickle.load(f1)

with open("saved/natural_reward_e004_May_iter3.data", "rb") as f1:
    rew_e004 = pickle.load(f1)

figure(figsize=(25, 5), dpi=180)
reward_e001 = np.sum(rew_e001[:270])
plt.plot(rew_e001[:270], 'g-', label='reward_e001, iter = 1')
# plt.plot(rew_e002[:270], 'b*-', label='reward_e002')
# plt.plot(rew_e003[:270], 'k--', label='reward_e003')
# plt.plot(rew_e004[:270], 'r-.', label='reward_e004')

plt.plot(rew_e004[270:270*2], 'b', label='reward_e001, iter = 2')

plt.plot(rew_e004[-270:], 'k', label='reward_e001, iter = 3')

plt.title("reward of E004, iter=3")
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

with open("saved/natural_reward_e003.data_May_iter3", "rb") as f1:
    rew_e003 = pickle.load(f1)

with open("saved/natural_reward_e004_May_iter3.data", "rb") as f1:
    rew_e004 = pickle.load(f1)

figure(figsize=(25, 5), dpi=180)
plt.plot(rew_e001[:], 'g-', label='reward_e001')
# plt.plot(rew_e002[:], 'b*', label='reward_e002')
# plt.plot(rew_e003[:], 'k--', label='reward_e003')
# plt.plot(rew_e004[:], 'r-.', label='reward_e004')

plt.xlabel("every 3 hours")
plt.legend()
plt.show()
"""

# """
# plots: dcdc, ac_in, ssr, wasted...
output_sum_May_default = "oist_summary_May_default.csv"
output_sum_May_default_2 = "oist_summary_May_default_2.csv"
oist_summary_May_defa_lin = "oist_summary_May_defa_lin.csv"

output_sum_May_iter1 = "oist_summary_May_iter1_3.csv"
output_sum_May_iter1_shuf = "oist_summary_May_iter1_shuffle.csv"
output_sum_May_iter3 = "oist_summary_May_iter3.csv"
output_sum_May_iter1_1hr = "oist_summary_May_iter1_1hr.csv"

oist_sum_May_prio_iter1_3hr = "oist_summary_May_Prior_iter1.csv"

output_getpath = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
output_dir = output_getpath + "/apis-emulator/data/output"

output_file_default = os.path.join(output_dir, output_sum_May_default)
# output_file_default_2 = os.path.join(output_dir, output_sum_May_default_2)
output_file_default_lin = os.path.join(output_dir, oist_summary_May_defa_lin)
output_file_iter1 = os.path.join(output_dir, output_sum_May_iter1)
output_file_iter1_shuf = os.path.join(output_dir, output_sum_May_iter1_shuf)
output_file_iter3 = os.path.join(output_dir, output_sum_May_iter3)
output_file_iter1_1hr = os.path.join(output_dir, output_sum_May_iter1_1hr)

output_file_prio_iter1_3hr = os.path.join(output_dir, oist_sum_May_prio_iter1_3hr)

data_default = pd.read_csv(output_file_default)
# data_default_2 = pd.read_csv(output_file_default_2)
data_default_lin = pd.read_csv(output_file_default_lin)
data_iter1 = pd.read_csv(output_file_iter1)
data_iter1_shuf = pd.read_csv(output_file_iter1_shuf)
data_iter3 = pd.read_csv(output_file_iter3)
data_iter1_1hr = pd.read_csv(output_file_iter1_1hr)

data_prio_iter1_3hr = pd.read_csv(output_file_prio_iter1_3hr)

dcdc_default = data_default['wg'][0:30]
# dcdc_default_2 = data_default_2['wg'][0:30]
dcdc_default_lin = data_default_lin['wg'][0:30]
dcdc_iter1 = data_iter1['wg'][0:30]
dcdc_iter1_shuf = data_iter1_shuf['wg'][0:30]
dcdc_iter3 = data_iter3['wg'][0:90]
dcdc_iter1_1hr = data_iter1_1hr['wg'][0:30]

dcdc_prio_iter1_3hr = data_prio_iter1_3hr['wg'][0:30]

acin_default = data_default['acin']  # lin
acin_iter1 = data_iter1_1hr['acin']

acin_prio_iter1 = data_prio_iter1_3hr['acin']

wasted_default = data_default['wasted']
wasted_iter1 = data_iter1['wasted']
wasted_prio_iter1 = data_prio_iter1_3hr['wasted']

ssr_pv_default = data_default['ssr_pv']
ssr_pv_iter1 = data_iter1['ssr_pv']
ssr_pv_prio_iter1 = data_prio_iter1_3hr['ssr_pv']

# bar plot of sum
acin_default_sum = acin_default[31]
acin_iter1_sum = acin_iter1[31]
acin_prio_iter1_sum = acin_prio_iter1[31]
wasted_default_sum = wasted_default[31]
wasted_iter1_sum = wasted_iter1[31]
wasted_prio_iter1_sum = wasted_prio_iter1[31]

data = [[acin_default_sum, wasted_default_sum],
[acin_iter1_sum, wasted_iter1_sum], [acin_prio_iter1_sum, wasted_prio_iter1_sum]]
X = np.arange(3)

# fig, ax = plt.subplots(1, 1)
# # ax2 = ax.twinx()
# # ax = fig.add_axes([0, 0, 1, 1])
# ax.bar(X + 0.00, data[0], width=0.25)
# ax.bar(X + 0.25, data[1], width=0.25)
# ax.legend(labels=['sum of default', 'sum of DQN'])
# # ax.set_ylabel('Power [W]')
# ax.set_title('purchased and wasted power [W]')
# plt.xticks([0.1, 1.1], ['purchased', 'wasted'])
figure(figsize=(15, 10), dpi=80)
# plt.plot(acin_default[0:30], 'g--', label='default purchased power')
# plt.plot(acin_iter1[0:30], 'r*-', label='DQN purchased power')
# plt.plot(acin_prio_iter1[0:30], 'b-', label='Prior_DQN purchased power')

# plt.plot(ssr_pv_default[0:30], 'g--+', label='default ssr')
# plt.plot(ssr_pv_iter1[0:30], 'r*-', label='DQN ssr')
# plt.plot(ssr_pv_prio_iter1[0:30], 'b*-', label='Prio_DQN ssr')

plt.plot(dcdc_default, 'g--', label='default exchanged power')
# plt.plot(dcdc_default_2, 'c--', label='default exchanged power macbook')
# plt.plot(dcdc_default_lin, 'm--', label='default exchanged power')
#
#####
plt.plot(dcdc_iter1, 'ro-', label='DQN exchanged power, iter=1, 3hrs')
# plt.plot(dcdc_iter1_shuf, 'b*-', label='DQN exchanged power, shuffle, iter=1, 3hrs')
####
# plt.plot(dcdc_iter3, 'go-', label='DQN exchanged power, iter=3')
# plt.plot(dcdc_iter1_1hr, 'k--', label='DQN exchanged power, iter=1, 1hr')

plt.plot(dcdc_prio_iter1_3hr, 'b-', label='Prio_DQN exchanged power, iter=1, 3hrs')
# #
# plt.xlabel("Days")
# plt.ylabel("Power [W]")
# # plt.ylabel("Rate")
# # plt.ylim(0, 1)
plt.legend(loc='upper right')
plt.show()
# """