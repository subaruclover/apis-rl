"""
raw data dcdc plots
(quarterhour data 4 houses)

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

# TODO: change to hourly data (keep the same as other methods)
output_ind_May_e001 = "house212_dcdc_min_May.csv"  # "house212_dcdc_hour_May.csv"
output_ind_May_e002 = "house213_dcdc_min_May.csv"  # "house213_dcdc_hour_May.csv"
output_ind_May_e003 = "house214_dcdc_min_May.csv"  # "house214_dcdc_hour_May.csv"
output_ind_May_e004 = "house215_dcdc_min_May.csv"  # "house215_dcdc_hour_May.csv"

output_getpath = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
output_dir = output_getpath + "/apis-emulator/data/input/Oist"

output_file_e001 = os.path.join(output_dir, output_ind_May_e001)
output_file_e002 = os.path.join(output_dir, output_ind_May_e002)
output_file_e003 = os.path.join(output_dir, output_ind_May_e003)
output_file_e004 = os.path.join(output_dir, output_ind_May_e004)

data_e001 = pd.read_csv(output_file_e001)
data_e002 = pd.read_csv(output_file_e002)
data_e003 = pd.read_csv(output_file_e003)
data_e004 = pd.read_csv(output_file_e004)

wg_e001 = data_e001['dcdc_grid_power'].values
wg_e002 = data_e002['dcdc_grid_power'].values
wg_e003 = data_e003['dcdc_grid_power'].values
wg_e004 = data_e004['dcdc_grid_power'].values


def get_Wh_data(N_DAYS, hour, wg_data):
    # output [Wh] data for every 24 hours (each day)
    # input: wg data for each agent, all days
    # output: [Wh] data for each house, each day

    wg_data_days, wg_Wh_hrs_charge, wg_Wh_hrs_discharge = [], [], []

    for i in range(N_DAYS):
        wg_data_day = wg_data[i * hour * 60: (i + 1) * hour * 60]
        wg_data_days.append(wg_data_day)

    for i in range(N_DAYS):
        for j in range(hour):
            wg_hr = wg_data_days[i][j * 60: (j + 1) * 60]
            # TODO: separate positive and negative values (charge/discharge)
            # set two variables for char/dischar respectively
            wg_hr_charge = np.sum(x*1/60 for x in wg_hr if x >= 0)
            wg_hr_discharge = np.sum(x*1/60 for x in wg_hr if x < 0)

            wg_Wh_hrs_charge.append(wg_hr_charge)
            wg_Wh_hrs_discharge.append(wg_hr_discharge)
            # reshape both char/dischar value, and return both values, plot in one figure
            # wg_hr = np.sum(wg_hr * 1 / 60)  # transfer [W] to [Wh]
            #
            # wg_Wh_hrs.append(wg_hr)

    # reshape to (N_DAYS, 24)
    # wg_Wh_hrs = np.reshape(wg_Wh_hrs, (N_DAYS, hour))

    return wg_Wh_hrs_charge, wg_Wh_hrs_discharge


N_DAYS = 31
hour = 24

# wg_Wh_hr_e001 = get_Wh_data(N_DAYS, hour, wg_e001)
# wg_Wh_hr_e002 = get_Wh_data(N_DAYS, hour, wg_e002)
# wg_Wh_hr_e003 = get_Wh_data(N_DAYS, hour, wg_e003)
# wg_Wh_hr_e004 = get_Wh_data(N_DAYS, hour, wg_e004)

wg_Wh_hr_char_e001, wg_Wh_hr_dischar_e001 = get_Wh_data(N_DAYS, hour, wg_e001)
wg_Wh_hr_char_e002, wg_Wh_hr_dischar_e002 = get_Wh_data(N_DAYS, hour, wg_e002)
wg_Wh_hr_char_e003, wg_Wh_hr_dischar_e003 = get_Wh_data(N_DAYS, hour, wg_e003)
wg_Wh_hr_char_e004, wg_Wh_hr_dischar_e004 = get_Wh_data(N_DAYS, hour, wg_e004)

# plot [Wh]
figure(figsize=(25, 5), dpi=180)
# plt.plot(wg_Wh_hr_e001, 'yo-', label='E001 dcdc')
# plt.plot(wg_Wh_hr_e002, 'm--', label='E002 dcdc')
# plt.plot(wg_Wh_hr_e003, 'g*-', label='E003 dcdc')
# plt.plot(wg_Wh_hr_e004, 'r-.', label='E004 dcdc')

plt.plot(wg_Wh_hr_char_e001, 'yo-', label='E001 dcdc')
plt.plot(wg_Wh_hr_char_e002, 'm--', label='E002 dcdc')
plt.plot(wg_Wh_hr_char_e003, 'g', label='E003 dcdc')
plt.plot(wg_Wh_hr_char_e004, 'r-.', label='E004 dcdc')

plt.plot(wg_Wh_hr_dischar_e001, 'y')
plt.plot(wg_Wh_hr_dischar_e002, 'm--')
plt.plot(wg_Wh_hr_dischar_e003, 'g*-')
plt.plot(wg_Wh_hr_dischar_e004, 'r-.')

plt.title('House 212, 213, 214, 215 raw dcdc power [Wh], by hour')
plt.legend(loc='upper right')
plt.show()

