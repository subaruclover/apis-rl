#!/usr/bin/env python
"""
DQN training, single run, house E002

created by: Qiong

"""
import pickle

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
# import tensorflow as tf
# print("tf version", tf.__version__)
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import logging.config

logger = logging.getLogger(__name__)

import time
import global_var as gl
import config as conf
import requests, json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# RL_learn functions
"""
class DQNNet : Deep Q-network Model 
class Memory : Memory model
class BatteryEnv: my battery model -> replaced with APIS battery model
"""
from RL_learn import Memory, DQNPrioritizedReplay
from agent import APIS, House

# agent = APIS()

# start_time = time.time()

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

############################
env = House(action_request=[7, 5], action_accept=[6])
env.seed(1)


MEMORY_SIZE = 10000  # 10000

sess = tf.Session()

with tf.variable_scope('natural_DQN'):
    RL_natural = DQNPrioritizedReplay(
        n_actions=8, n_features=6, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.008, sess=sess, prioritized=False, output_graph=True,
    )


# with tf.variable_scope('DQN_with_prioritized_replay'):
#     RL_prio = DQNPrioritizedReplay(
#         n_actions=8, n_features=6, memory_size=MEMORY_SIZE,
#         e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
#     )

sess.run(tf.global_variables_initializer())


def train(RL):
    print("House E002, training start")
    total_steps = 0
    steps = []
    episodes = []
    reward_list = []
    EPI = 1
    N_DAY = 30

    # house_id = input('input the house id: ')

    for i_episode in range(EPI):

        print("Episode {} starts".format(i_episode))
        day = 0
        hour = 0
        done = False

        # TODO: agent needs to get value from the env, not given
        # reset with the env?
        observation = env.reset(house_id)
        start_time = time.time()

        while day < N_DAY:  # not gl.sema:

            actions = RL.choose_actions(observation)
            action_request = [actions[0], actions[2]]
            action_accept = [actions[1]]

            # agent.CreateSce2(action_request, action_accept)

            # house_id = input('input the house id: ')
            observation_, reward, info = env.step2(action_request, action_accept, house_id)

            actions_space = np.linspace(0.2, 0.9, 8).tolist()
            # actions_space = np.around(np.linspace(0.3, 0.9, 7).tolist(), 1)
            print("House E002, Scenario file updated with act_req {}, {} and act_acc {}".format(actions_space[action_request[0]],
                                                                                  actions_space[action_request[1]],
                                                                                  actions_space[action_accept[0]]))

            # change the time step
            # time.sleep(60)

            # if time.sleep(5):  # done:
            #     reward = p2_e001

            RL.store_transition(observation, actions, reward, observation_)

            reward_list.append(reward)

            # if total_steps > MEMORY_SIZE:
            if (total_steps > 24*3) and (total_steps % 2 == 0):
                RL.learn()

            if hour < 24/3:  # 24 - 1:#(total_steps > 0) and (total_steps % 24 == 0):  # one day
                hour += 1
                observation = observation_
                total_steps += 1
                print("total_steps = ", total_steps)
                time.sleep(60*3)  # update every hour
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
                    break

            # observation = observation_
            # total_steps += 1

        # end_time = time.time()
        # print("episode {} - training time: {:.2f}mins".format(i_episode, (end_time - start_time) / 60 * gl.acc))

    # return np.vstack((episodes, steps)), RL.memory
    return RL.memory, reward_list


house_id = "E002"  # input('input the house id: ')
# his_natural, natural_memory = train(RL_natural)
natural_memory, natural_reward = train(RL_natural)
# his_prio, prio_memory = train(RL_prio)
# prio_memory_store = [prio_memory.tree.data[i][9] for i in range(24*55)]  # reward(p2)
#  save memo to json file
with open("saved/natural_memo_e002.data", "wb") as fp:
    pickle.dump(natural_memory, fp)
#  save reward to json file
with open("saved/natural_reward_e002.data", "wb") as fp:
    pickle.dump(natural_reward, fp)

# with open("saved/prio_memo_e002.data", "wb") as fp:
#     pickle.dump(prio_memory, fp)
# #  save reward to json file
# with open("saved/prio_reward_e002.data", "wb") as fp:
#     pickle.dump(prio_memory_store, fp)

# compare based on first success
# plt.title("E002")
# plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='g', label='natural DQN')
# plt.plot(natural_memory[:24, 8], 'b', label='natural DQN p2')
# plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='b', label='DQN with prioritized replay')
# plt.plot(prio_memory_store, 'g', label='DQN with prioritized replay')
# plt.legend(loc='best')
# plt.ylabel('reward (p2)')
# plt.xlabel('episode (hour)')
# plt.grid()
# plt.show()
