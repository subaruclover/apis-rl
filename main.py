#!/usr/bin/env python
"""
DQN training, single run, house E001

created by: Qiong

TODO: done condition:
Check if this could work:
although the APIS emulator is an online version simulator,
can we use an offline RL (update its policy only) method?
- No, there is no such a function.

i.e. load the data here (from apis-emulator/data/input/Sample directory),
choose the data we would use (e.g. E001~E004, one year, 2020/4/1 ~ 2021/3/31, or some other time period)
set 24 data points for each day, and update their RSOCs with SonyCSL's APIS

Note that sample data have 48 data points each day (record every 30mins), we only need 24 for testing

"""
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import logging.config

logger = logging.getLogger(__name__)
import time
import global_var as gl
import analyser
import core
import config as conf
import requests, json
import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
import h5py

# RL_learn functions
"""
class DQNNet : Deep Q-network Model -> redefined in DQNPrioritizedReplay, therefore no need anymore
class Memory : Memory model
class BatteryEnv: my battery model -> replaced with APIS battery model
"""

from RL_learn import Memory, DQNPrioritizedReplay
from agent import APIS, House  # Env

# agent = APIS()

# start_time = time.time()
"""
##############################
# Data loading
# get log data for states
host = conf.b_host
port = conf.b_port
# url = "http://0.0.0.0:4390/get/log"
URL = "http://" + host + ":" + str(port) + "/get/log"


# dicts of states for all houses
pvc_charge_power = {}
ups_output_power = {}
p2 = {}  # powermeter.p2, Power consumption to the power storage system [W]
rsoc = {}
wg = {}  # meter.wg, DC Grid power [W]
wb = {}  # meter.wb, Battery Power [W]

pv_list = []
load_list = []
p2_list = []
"""
# need to refresh the output data every 5s? time.sleep()
"""
while gl.sema:  # True, alter for different time periods
    # # refresh every 5 seconds
    # time.sleep(5)
    # read variables from /get/log url
    # print(output_data.text)
    output_data = requests.get(URL).text
    output_data = json.loads(output_data)  # dict

    rsoc_list = []

    for ids, dict_ in output_data.items():  # ids: E001, E002, ... house ID
        # print('the name of the dictionary is ', ids)
        # print('the dictionary is ', dict_)
        # when ids is "E001" (change to other house ID for other houses)
        pvc_charge_power[ids] = output_data[ids]["emu"]["pvc_charge_power"]
        ups_output_power[ids] = output_data[ids]["emu"]["ups_output_power"]
        p2[ids] = output_data[ids]["dcdc"]["powermeter"]["p2"]
        rsoc[ids] = output_data[ids]["emu"]["rsoc"]
        wg[ids] = output_data[ids]["dcdc"]["meter"]["wg"]
        wb[ids] = output_data[ids]["dcdc"]["meter"]["wb"]

        print("pv of {ids} is {pv},".format(ids=ids, pv=pvc_charge_power[ids]),
              "load of {ids} is {load},".format(ids=ids, load=ups_output_power[ids]),
              "p2 of {ids} is {p2},".format(ids=ids, p2=p2[ids]),
              "rsoc of {ids} is {rsoc},".format(ids=ids, rsoc=rsoc[ids])
              # "wg of {ids} is {wg},".format(ids=ids, wg=wg[ids]),
              # "wb of {ids} is {wb},".format(ids=ids, wb=wb[ids])
              )
        rsoc_list.append(rsoc[ids])
        # refresh every 5 seconds
        # print("\n")
        # time.sleep(5)

        # States  pvc_charge_power[ids], for house E001
        if ids == "E001":
            pv_e001 = np.array([pvc_charge_power["E001"]])
            load_e001 = np.array([ups_output_power["E001"]])
            p2_e001 = np.array([p2["E001"]])
            rsoc_e001 = np.array([rsoc["E001"]])

            x_e001 = np.concatenate([pv_e001, load_e001, p2_e001, rsoc_e001], axis=-1)
            print(x_e001)  # [39.14 575.58 734.    29.98] E001
    ##
    # print(rsoc)
    # {'E001': 29.98, 'E002': 29.99, 'E003': 29.98, 'E004': 29.99}
    rsoc_ave = np.mean(rsoc_list)  # get average rsoc of this community
    # print(rsoc_ave)
    # state = np.concatenate((x_e001, rsoc_ave), axis=-1)

    state_size = (5,)
    action_request_space = np.linspace(0.2, 0.9, 8).tolist()  # [0.2~0.9], 8 options
    action_accept_space = np.linspace(0.2, 0.9, 8).tolist()
    action_request_num = len(action_request_space)
    action_accept_num = len(action_accept_space)
    learning_rate = 0.01
    # action_request = sorted(np.random.randint(0, action_request_num, 2), reverse=True)  # 2 values
    # action_accept = np.random.randint(0, action_accept_num, 1)
    # actions_request = sorted(random.sample(action_request_space, 2))  # 2 values
    # actions_accept = random.sample(action_request_space, 1)  # 1 value
    action_request = sorted(np.random.choice(action_request_num, 2, replace=False), reverse=True)  # 2 values
    action_accept = np.random.randint(action_request[1], action_request[0], 1)  # 1 value between 2 request actions

    # agent.CreateSce(action_request, action_accept)

    # Training hyperparameters
    batch_size = 256
    # EPI = 10

    # Exploration hyperparameters for epsilon greedy strategy
    explore_start = 1.0  # exploration probability at start
    explore_stop = 0.01  # minimum exploration probability
    decay_rate = 0.001  # exponential decay rate for exploration prob

    # Q-learning hyperparameters
    gamma = 0.96  # Discounting rate of future reward

    # Memory hyperparameters
    pretrain_length = 10000  # # of experiences stored in Memory during initialization
    memory_size = 10000  # # of experiences Memory can keep

    # TODO: battery update replaced
    # battery = BatteryEnv(action_size=action_size)
    # how the battery changes: from APIS
    # action: scenario generation variables (request, accept, etc..)
    # action refresh to create new scenarios
    # if rsoc[ids]

    memory = Memory(memory_size)

    np.random.seed(42)

    # Memory initialization
    day = 0
    quarter_hour = 0
    done = False
    # timestep = 15.0

    state = x_e001

    # Compute the reward and new state based on the selected action
    # next_rsoc, batteryLevel, reward
    # batteryLevel_req, batteryLevel_acc = agent.step(action_request, action_accept)
    # batteryLevel = agent.step(state, action_request, action_accept)
    agent.CreateSce1(action_request, action_accept)

    print("req_act: ", action_request_space[action_request[0]], action_request_space[action_request[1]],
          "acc_act: ", action_accept_space[action_accept[0]])
    time.sleep(60)  # 5s
"""

############################
# give a set of init actions (action 5, 6, 7 for act_req and act_acc)
# env = House(action_request=[7, 5], action_accept=[6])
# env.seed(1)


def combine_actions(RL, observation):
    # Do action combination? with \theta probability
    # choose basic a1 and a2 using softmax/greedy..
    # create a new action a_combine
    # add a_combine to Action set A
    # restrictions of actions?
    action_request = RL.choose_action(observation)  # need 2
    action_accept = RL.choose_action(observation)

    combine_action = np.array([action_request, action_accept])

    return combine_action


# TODO: separate data into training set (5 month) + testing set


def train(RL):
    print("House E001, training start")

    total_steps = 0
    steps = []
    episodes = []
    reward_list = []
    EPI = 3  # #.of iter
    # N_RUN = 3  # 4
    N_DAY = 30  # 30

    # give a set of init actions (action 5, 6, 7 for act_req and act_acc)
    env = House(action_request=[7, 5], action_accept=[6])
    env.seed(1)

    # for i_run in range(N_RUN):
    #     print("********Run {} starts********".format(i_run))
    #     total_reward = 0
    #     # day = 0
    #     # hour = 0
    #     # done = False

    for i_episode in range(EPI):
        # 1 EPI: test one month data first (shorten from one year)
        # (1day, 24hrs) 24 min, action updated every hour (1 min)
        print("********Episode {} starts*********".format(i_episode))
        day = 0
        hour = 0
        done = False

        # TODO: (when reset) agent needs to get value from the env, not given
        # reset with the env
        observation = env.reset_time(house_id)
        total_reward = 0

        while day < N_DAY:  # True:  # not gl.sema: total_steps <= 24 (one day)

            # start_time = time.time()

            # choose actions (e-greedy)
            actions = RL.choose_actions(observation)
            action_request = [actions[0], actions[2]]
            action_accept = [actions[1]]

            # house_id = input('input the house id: ')
            # TODO: add done (how to make it offline? with the current online simulation)
            observation_, reward, info = env.step1_time(action_request, action_accept, house_id)

            # actions_space = np.around(np.linspace(0.3, 0.9, 7).tolist(), 1)
            actions_space = np.linspace(0.2, 0.9, 8).tolist()
            print("House E001, Scenario file updated with act_req {}, {} and act_acc {}".format(
                actions_space[action_request[0]],
                actions_space[action_request[1]],
                actions_space[action_accept[0]]))
            # Store the experience in memory
            RL.store_transition(observation, actions, reward, observation_)

            # reward_list.append(reward)
            total_reward += reward
            # print("total step", total_steps)
            # if total_steps > 100:  # MEMORY_SIZE
            #     RL.learn()
            # start learn after 100 steps and the frequency of learning
            # accumulate some memory before start learning
            if total_steps > 24 * 3:  # and (total_steps % 2 == 0):
                RL.learn()
            # RL.learn()

            if hour < 24 / 3:  # 24 - 1:#(total_steps > 0) and (total_steps % 24 == 0):  # one day
                hour += 1
                observation = observation_
                total_steps += 1
                print("total_steps = ", total_steps)

                time.sleep(60*3)  # update every 3 hours
            else:
                done = True
                day += 1
                print('Day', day, ' finished')
                hour = 0

                if day < N_DAY:
                    observation = observation_
                    steps.append(total_steps)
                    episodes.append(i_episode)
                else:
                    # print('********End of input Days.********')
                    break

            # observation = observation_
            # total_steps += 1
            # print("total_steps = ", total_steps)
        # Track rewards
        reward_list.append(total_reward)
        # print(reward_list)
        # end_time = time.time()
        # print("episode {} - training time: {:.2f}mins".format(i_episode, (end_time - start_time) / 60 * gl.acc))

    # save trained model
    saver = tf.train.Saver()
    saver.save(RL.sess, 'model/E001/E001_model_prio')
    print('Model Trained and Saved')

    # print(reward_list)

    # return np.vstack((episodes, steps)), RL.memory
    return RL.memory, reward_list


house_id = "E001"  # input('input the house id: ')
# his_natural, natural_memory = train(RL_natural)
# natural_memory, natural_reward = train(RL_natural)
# # natural_memory_store = [natural_memory.tree.data[i][9] for i in range(24*55)]  # reward(p2)
# #  save memo to json file
# with open("saved/natural_memo_e001_May_iter1_time.data", "wb") as fp:
#     pickle.dump(natural_memory, fp)
# #  save reward to json file
# with open("saved/natural_reward_e001_May_iter1_time.data", "wb") as fp:
#     pickle.dump(natural_reward, fp)

##

MEMORY_SIZE = 10000  # 10000

sess = tf.Session()

# EPI = 1, e_greedy_increment=0.01
# EPI = 3, e_greedy_increment=0.005

# with tf.variable_scope('natural_DQN'):
#     RL_natural = DQNPrioritizedReplay(
#         n_actions=8, n_features=8, memory_size=MEMORY_SIZE,
#         e_greedy_increment=0.008, sess=sess, prioritized=False, output_graph=True,
#     )

with tf.variable_scope('DQN_with_prioritized_replay', reuse=tf.AUTO_REUSE):
    RL_prio = DQNPrioritizedReplay(
        n_actions=8, n_features=8, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.0015, seed=1, sess=sess, prioritized=True, test=False, output_graph=True,
    )  # n_features: 6 states

sess.run(tf.global_variables_initializer())

# his_prio, prio_memory = train(RL_prio)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int)

    args = parser.parse_args()

    prio_memory, prio_reward = train(RL_prio)
# prio_memory_store = [prio_memory.tree.data[i][9] for i in range(24*55)]  # reward(p2)
# save memo to json file
# with open("saved/prio_memo_e001_May_train_time.data", "wb") as fp:
#     pickle.dump(prio_memory, fp)
# # save reward to json file
# with open("saved/prio_reward_e001_May_train_time.data", "wb") as fp:
#     pickle.dump(prio_reward, fp)

# saver.save(sess, 'E001_model')



