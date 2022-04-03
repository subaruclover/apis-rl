"""
analyze actions from
1) saved data (from ./saved/*.data)
2) indivLog*.csv files (from apis-emulator/data/output)

take apis-service_center for reference

Create by Qiong
"""

import os
import pandas as pd
import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

output_ind_May_default = "oist_indivLog_May_default.csv"
# output_ind_May_default_2 = "oist_indivLog_May_default_2.csv"
output_ind_May_iter1 = "oist_indivLog_May_iter1.csv"
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

wg_e001 = dcdc_default_e001['dcdc_meter_wg'].values
wg_e002 = dcdc_default_e002['dcdc_meter_wg'].values
wg_e003 = dcdc_default_e003['dcdc_meter_wg'].values
wg_e004 = dcdc_default_e004['dcdc_meter_wg'].values


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
            wg_hr = np.sum(wg_hr * 1 / 60)  # transfer [W] to [Wh]

            wg_Wh_hrs.append(wg_hr)

    wg_Wh_hrs = np.reshape(wg_Wh_hrs, (N_DAYS, hour))

    return wg_Wh_hrs


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

wg_Wh_hr_e001 = get_Wh_data(N_DAYS, hour, wg_e001)
wg_Wh_hr_e002 = get_Wh_data(N_DAYS, hour, wg_e002)
wg_Wh_hr_e003 = get_Wh_data(N_DAYS, hour, wg_e003)
wg_Wh_hr_e004 = get_Wh_data(N_DAYS, hour, wg_e004)
# plot functions (for multiple subplots)


# hourly (data per min: each hour has 60 points; 1 Day = 1440 points)
# fig, ax = plt.subplots(1, 1)
# X = np.linspace(0, 23, 24)
# plt.bar(X, wg_days_e003[0], width=0.25)


# plt.plot(dcdc_default_e001['dcdc_meter_wg'])

x_axis = np.linspace(0, 23, 24)
days = np.linspace(0, N_DAYS-2, N_DAYS-1)  # plot 30 days

fig = plt.figure(figsize=(24, 30))

for idx, i in enumerate(days.astype(int)):
    ax = fig.add_subplot(5, 6, idx+1)
    ax.bar(x_axis, wg_Wh_hr_e003[i], width=0.5)
    ax.set_xlabel("hour")
    ax.set_ylabel("amount [Wh]")

fig.suptitle("Charge/Discharge Wh, E003")
# plt.bar(x_axis, wg_Wh_hr_e003[0], width=0.5)
# plt.xlabel('hour')
# plt.ylabel('amount [Wh]')
plt.show()

