"""
analyze the daily output log data (from apis-emulator/data/output/daily_output.txt)

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

output_filename = "daily_output.txt"

output_getpath = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
output_dir = output_getpath + "/apis-emulator/data/output"

output_file = os.path.join(output_dir, output_filename)

# with open(output_file) as f:
#     contents = f.readlines()

data = pd.read_csv(output_file, header=None)

days_wg = []

for i in range(1, len(data)):
    daily_wg = float(data.iloc[i][9])
    days_wg.append(daily_wg)

# check the dcdc data in running days
plt.plot(days_wg, 'go-', label="daily cumulative dcdc power [W]")
plt.xlabel("Day")
plt.legend()
plt.show()