"""
analyze p2 data from
1) saved data (from ./saved/*.data)
2) indivLog*.csv files (from apis-emulator/data/output)

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

output_ind_May_default = "oist_indivLog_May_default.csv"
# output_ind_May_default = "oist_indivLog_May_defa_lin.csv"

# output_ind_May_default_2 = "oist_indivLog_May_default_2.csv"
# output_ind_May_iter1 = "oist_indivLog_May_iter1.csv"
output_ind_May_iter1 = "oist_indivLog_May_Prior_iter5.csv"  # "oist_indivLog_May_iter1_1hr.csv"  # "oist_indivLog_May_iter1_3.csv"
# oist_summary_May_iter1_1hr
# output_ind_May_iter3 = "oist_indivLog_May_iter3.csv"


output_getpath = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
output_dir = output_getpath + "/apis-emulator/data/output"

output_file_default = os.path.join(output_dir, output_ind_May_default)
# output_file_default_2 = os.path.join(output_dir, output_ind_May_default_2)
output_file_iter1 = os.path.join(output_dir, output_ind_May_iter1)
# output_file_iter3 = os.path.join(output_dir, output_ind_May_iter3)

data_default = pd.read_csv(output_file_default)
# data_default_2 = pd.read_csv(output_file_default_2)
data_iter1 = pd.read_csv(output_file_iter1)
# data_iter3 = pd.read_csv(output_file_iter3)

dcdc_default_e001 = data_default.loc[data_default['id'] == 'E001']
dcdc_default_e002 = data_default.loc[data_default['id'] == 'E002']
dcdc_default_e003 = data_default.loc[data_default['id'] == 'E003']
dcdc_default_e004 = data_default.loc[data_default['id'] == 'E004']

dcdc_iter1_e001 = data_iter1.loc[data_iter1['id'] == 'E001']
dcdc_iter1_e002 = data_iter1.loc[data_iter1['id'] == 'E002']
dcdc_iter1_e003 = data_iter1.loc[data_iter1['id'] == 'E003']
dcdc_iter1_e004 = data_iter1.loc[data_iter1['id'] == 'E004']

# default values (p2)
p2_defa_e001 = dcdc_default_e001['dcdc_powermeter_p2'].values
p2_defa_e002 = dcdc_default_e002['dcdc_powermeter_p2'].values
p2_defa_e003 = dcdc_default_e003['dcdc_powermeter_p2'].values
p2_defa_e004 = dcdc_default_e004['dcdc_powermeter_p2'].values

###
# RL learners' values (p2)
p2_e001 = dcdc_iter1_e001['dcdc_powermeter_p2'].values
p2_e002 = dcdc_iter1_e002['dcdc_powermeter_p2'].values
p2_e003 = dcdc_iter1_e003['dcdc_powermeter_p2'].values
p2_e004 = dcdc_iter1_e004['dcdc_powermeter_p2'].values


def trim_wg(wg_data):
    # get grid power by days
    # input: wg data for each agent, all days
    # output: wg data for each house, each day

    N_DAYS = 31
    wg_data_days = []

    for i in range(N_DAYS):
        wg_data_day = wg_data[i * 1440: (i + 1) * 1440]
        wg_data_days.append(wg_data_day)
        # i += 1

    return wg_data_days


def cal_wg_hr(wg_days):
    # calculate grid power by hour
    # input: wg data by day (1440 points/mins)
    # output: wg data by hour

    N_DAYS = 31
    hour = 24
    wg_hrs = []

    for i in range(N_DAYS):
        for j in range(hour):
            wg_hr = wg_days[i][j * 60:(j + 1) * 60]
            wg_hr = np.sum(wg_hr * 1 / 60)  # transfer [W] to [Wh]
            wg_hrs.append(wg_hr)
            j += 1
        i += 1

    return wg_hrs


def get_Wh_data(N_DAYS, hour, wg_data):
    # output [Wh] data for every 24 hours (each day)
    # input: wg data for each agent, all days
    # output: [Wh] data for each house, each day

    wg_data_days, wg_Wh_hrs = [], []

    for i in range(N_DAYS):
        wg_data_day = wg_data[i * hour * 60: (i + 1) * hour * 60]
        wg_data_days.append(wg_data_day)

    for i in range(N_DAYS):
        for j in range(hour):
            wg_hr = wg_data_days[i][j * 60: (j + 1) * 60]
            # TODO: separate positive and negative values (charge/discharge)
            # set two variables for char/dischar respectively
            # wg_hr_charge = np.sum(wg_hr*1/60) if wg_hr >= 0
            # wg_hr_discharge = np.sum(wg_hr*1/60) if wg_hr < 0
            # wg_Wh_hrs_charge.append(wg_hr_charge)
            # wg_Wh_hrs_discharge.append(wg_hr_discharge)
            # reshape both char/dischar value, and return both values, plot in one figure
            wg_hr = np.sum(wg_hr * 1 / 60)  # transfer [W] to [Wh]

            wg_Wh_hrs.append(wg_hr)

    # reshape to (N_DAYS, 24)
    # wg_Wh_hrs = np.reshape(wg_Wh_hrs, (N_DAYS, hour))

    return wg_Wh_hrs


# plot functions (for multiple subplots)
# TODO: put charge/discharge data of all houses in one bar plot (by unit)
def plot_deal(wg_Wh_hr_data, N_DAYS, houseID):
    x_axis = np.linspace(0, 23, 24)
    days = np.linspace(0, N_DAYS - 2, N_DAYS - 1)  # plot 30 days

    fig = plt.figure(figsize=(24, 30))

    for idx, i in enumerate(days.astype(int)):
        ax = fig.add_subplot(5, 6, idx + 1)
        ax.bar(x_axis, wg_Wh_hr_data[i], width=0.5)
        ax.set_xlabel("hour")
        ax.set_ylabel("amount [Wh]")

    fig.suptitle("Charge/Discharge Wh, {}".format(houseID))

    plt.show()


# wg_days_e001 = trim_wg(wg_data=wg_e001)
# wg_days_e002 = trim_wg(wg_data=wg_e002)
# wg_days_e003 = trim_wg(wg_data=wg_e003)
# wg_days_e004 = trim_wg(wg_data=wg_e004)
#
# # wg_sum_hr_e001 = wg_days_e001
# wg_hr_e001 = cal_wg_hr(wg_days_e001)
# wg_hr_e002 = cal_wg_hr(wg_days_e002)
# wg_hr_e003 = cal_wg_hr(wg_days_e003)
# wg_hr_e004 = cal_wg_hr(wg_days_e004)
N_DAYS = 31
hour = 24

# calculate bought external power (p2) data (in [Wh])
# default
p2_defa_Wh_hr_e001 = get_Wh_data(N_DAYS, hour, p2_defa_e001)
p2_defa_Wh_hr_e002 = get_Wh_data(N_DAYS, hour, p2_defa_e002)
p2_defa_Wh_hr_e003 = get_Wh_data(N_DAYS, hour, p2_defa_e003)
p2_defa_Wh_hr_e004 = get_Wh_data(N_DAYS, hour, p2_defa_e004)

# DQN learner
p2_Wh_hr_e001 = get_Wh_data(N_DAYS, hour, p2_e001)
p2_Wh_hr_e002 = get_Wh_data(N_DAYS, hour, p2_e002)
p2_Wh_hr_e003 = get_Wh_data(N_DAYS, hour, p2_e003)
p2_Wh_hr_e004 = get_Wh_data(N_DAYS, hour, p2_e004)


# plot deal data for each house
# plot_deal(wg_Wh_hr_e001, N_DAYS, "E001")
# plot_deal(wg_Wh_hr_e002, N_DAYS, "E002")
# plot_deal(wg_Wh_hr_e003, N_DAYS, "E003")
# plot_deal(wg_Wh_hr_e004, N_DAYS, "E004")

# plt.plot(wg_Wh_hr_data, label='houseID')

fig, axs = plt.subplots(4, 1, figsize=(25, 20), dpi=180)
# plt.figure(1, figsize=(25, 5), dpi=180)
axs[0].plot(p2_defa_Wh_hr_e001, 'g-', label='E001 p2, default')
axs[0].plot(p2_Wh_hr_e001, 'r--', label='E001 p2, DQN')
axs[0].title.set_text('E001 p2 values [Wh] each day, by hour')
# axs[0].set_xlabel('hour')

# plt.subplots(4, 1, 2, figsize=(25, 5), dpi=180)
# plt.figure(2, figsize=(25, 5), dpi=180)
axs[1].plot(p2_defa_Wh_hr_e002, 'g-', label='E002 p2, default')
axs[1].plot(p2_Wh_hr_e002, 'r--', label='E002 p2, DQN')
axs[1].title.set_text('E002 p2 values [Wh] each day, by hour')

# plt.subplots(4, 1, 3, figsize=(25, 5), dpi=180)
# plt.figure(3, figsize=(25, 5), dpi=180)
axs[2].plot(p2_defa_Wh_hr_e003, 'g-', label='E003 p2, default')
axs[2].plot(p2_Wh_hr_e003, 'r--', label='E003 p2, DQN')
axs[2].title.set_text('E003 p2 values [Wh] each day, by hour')

# #
# plt.subplots(4, 1, 4, figsize=(25, 5), dpi=180)
# plt.figure(4, figsize=(25, 5), dpi=180)
axs[3].plot(p2_defa_Wh_hr_e004, 'g-', label='E004 p2, default')
axs[3].plot(p2_Wh_hr_e004, 'r--', label='E004 p2, DQN')
axs[3].title.set_text('E002 p2 values [Wh] each day, by hour')
# axs[3].set_xlabel('hour')

# plt.title('E001~E004 p2 values [Wh] each day, default scenario, by hour')
# plt.title('E001~E004 p2 values [Wh] each day, DQN scenario, iter=1, by hour')

# fig.supxlabel('hour')  # not working with python<3.7 and matplotlib<3.4
# fig.supylabel('p2 in [Wh]')
fig.text(0.5, 0.08, 'hour', ha='center', fontsize=22)
fig.text(0.08, 0.5, 'p2 in [Wh]', va='center', rotation='vertical', fontsize=22)

axs[0].legend(loc='upper right')
axs[1].legend(loc='upper right')
axs[2].legend(loc='upper right')
axs[3].legend(loc='upper right')
plt.gca().set_aspect('auto')
plt.show()
