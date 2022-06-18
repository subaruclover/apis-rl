#!/usr/bin/env python
"""
run the main3.py file for multiple times (runs)

Author: Qiong
"""

# import os
import pickle

N_RUN = 3
i_run = 0

while i_run < N_RUN:
    # execute the code for N_RUN times

    print("********Run {} starts********".format(i_run))
    exec(open("main3.py").read())
    # execfile('main3.py')
    # os.system("python main3.py")
    # f = os.popen("python main3.py")
    # main3.prio_memory
    # prio_memory, prio_reward = main3.train(main3.RL_prio)
    # print(prio_reward)

    with open("saved/prio_memo_e003_May_train_time_run{}.data".format(i_run), "wb") as fp:
        pickle.dump(prio_memory, fp)
    # save reward to json file
    with open("saved/prio_reward_e003_May_train_time_run{}.data".format(i_run), "wb") as fp:
        pickle.dump(prio_reward, fp)

    i_run += 1
