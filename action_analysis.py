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



# plt.plot(dcdc_default_e001['dcdc_meter_wg'])
# plt.show()
# ['dcdc_meter_wg'][0:30]

