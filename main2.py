#!/usr/bin/env python
"""
DQN training, single run, house E002

created by: Qiong

"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
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
from RL_learn import  Memory, DQNPrioritizedReplay
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
env.seed(21)

MEMORY_SIZE = 100  # 10000

sess = tf.Session()
with tf.variable_scope('natural_DQN'):
    RL_natural = DQNPrioritizedReplay(
        n_actions=8, n_features=5, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=False, output_graph=True,
    )


# with tf.variable_scope('DQN_with_prioritized_replay'):
#     RL_prio = DQNPrioritizedReplay(
#         n_actions=8, n_features=5, memory_size=MEMORY_SIZE,
#         e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
#     )

sess.run(tf.global_variables_initializer())


def train(RL):
    print("House E002, training start")
    total_steps = 0
    steps = []
    episodes = []
    # EPI = 15

    # house_id = input('input the house id: ')

    for i_episode in range(24):

        # TODO: agent needs to get value from the env, not given
        # reset with the env?
        observation = env.reset(house_id)
        start_time = time.time()

        while True:  # not gl.sema:

            actions = RL.choose_actions(observation)
            action_request = [actions[0], actions[2]]
            action_accept = [actions[1]]

            # agent.CreateSce2(action_request, action_accept)

            # house_id = input('input the house id: ')
            observation_, reward, done, info = env.step2(action_request, action_accept, house_id)

            actions_space = np.linspace(0.2, 0.9, 8).tolist()
            print("House E002, Scenario file updated with act_req {}, {} and act_acc {}".format(actions_space[action_request[0]],
                                                                                  actions_space[action_request[1]],
                                                                                  actions_space[action_accept[0]]))

            # change the time step
            # time.sleep(60)

            # if time.sleep(5):  # done:
            #     reward = p2_e001

            RL.store_transition(observation, actions, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break  #

            observation = observation_
            total_steps += 1

        end_time = time.time()
        print("episode {} - training time: {:.2f}mins".format(i_episode, (end_time - start_time) / 60 * gl.acc))

    return np.vstack((episodes, steps)), RL.memory


house_id = "E002"  # input('input the house id: ')
his_natural, natural_memory = train(RL_natural)
# his_prio, prio_memory = train(RL_prio)

# prio_memory_store = [prio_memory.tree.data[i][8] for i in range(24)]  # reward(p2)

# compare based on first success
plt.title("E002")
plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='g', label='natural DQN')
plt.plot(natural_memory[:24, 8], 'b', label='natural DQN p2')
# plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
# plt.plot(prio_memory_store, 'r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('reward (p2)')
plt.xlabel('episode')
plt.grid()
plt.show()
