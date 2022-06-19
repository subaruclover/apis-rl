#!/usr/bin/env python
"""
run the main.py file for multiple times (runs)

Author: Qiong
"""

# import os
import pickle
import sys

# import main as main

N_RUN = 3
i_run = 0

while i_run < N_RUN:
    # execute the code for N_RUN times

    print("********Run {} starts********".format(i_run))

    if i_run == 0:
        sys.argv = ['main.py', '--seed=1']
    elif i_run == 1:
        sys.argv = ['main.py', '--seed=21']
    elif i_run == 2:
        sys.argv = ['main.py', '--seed=42']

    exec(open("main.py").read())
    # execfile('main.py')  # removed in python 3.x
    # os.system("python main.py")
    # f = os.popen("python main.py")

    # print(prio_reward)

    with open("saved/prio_memo_e001_May_train_time_run{}.data".format(i_run), "wb") as fp:
        pickle.dump(prio_memory, fp)
    # save reward to json file
    with open("saved/prio_reward_e001_May_train_time_run{}.data".format(i_run), "wb") as fp:
        pickle.dump(prio_reward, fp)

    i_run += 1