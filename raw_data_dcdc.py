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

sns.set(style="whitegrid")

# TODO: change to hourly data (keep the same as other methods)
output_ind_May_e001 = "house212_dcdc_quarterhour_May.csv"
output_ind_May_e002 = "house213_dcdc_quarterhour_May.csv"
output_ind_May_e003 = "house214_dcdc_quarterhour_May.csv"
output_ind_May_e004 = "house215_dcdc_quarterhour_May.csv"

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

plt.plot(wg_e001, 'y-', label='E001 dcdc')
plt.plot(wg_e002, 'm*-', label='E002 dcdc')
plt.plot(wg_e003, 'g--', label='E003 dcdc')
plt.plot(wg_e004, 'bo-', label='E004 dcdc')

plt.title('House 212, 213, 214, 215 raw dcdc value [W] in quarter-hour')
plt.legend(loc='upper right')
plt.show()

