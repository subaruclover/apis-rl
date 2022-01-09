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

agent = APIS()

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

# need to refresh the output data every 5s? time.sleep()
while gl.sema:  # True, alter for different time periods
    # # refresh every 5 seconds
    # time.sleep(5)
    # read variables from /get/log url
    # print(output_data.text)
    output_data = requests.get(URL).text
    output_data = json.loads(output_data)  # dict

    for ids, dict_ in output_data.items():  # ids: E001, E002, ... house ID
        # print('the name of the dictionary is ', ids)
        # print('the dictionary is ', dict_)
        # when ids is "E002" (change to other house ID for other houses)
        pvc_charge_power[ids] = output_data[ids]["emu"]["pvc_charge_power"]
        ups_output_power[ids] = output_data[ids]["emu"]["ups_output_power"]
        p2[ids] = output_data[ids]["dcdc"]["powermeter"]["p2"]
        rsoc[ids] = output_data[ids]["emu"]["rsoc"]
        wg[ids] = output_data[ids]["dcdc"]["meter"]["wg"]
        wb[ids] = output_data[ids]["dcdc"]["meter"]["wb"]

        print("pv of {ids} is {pv},".format(ids=ids, pv=pvc_charge_power[ids]),
              "load of {ids} is {load},".format(ids=ids, load=ups_output_power[ids]),
              "p2 of {ids} is {p2},".format(ids=ids, p2=p2[ids]),
              "rsoc of {ids} is {rsoc},".format(ids=ids, rsoc=rsoc[ids]),
              "wg of {ids} is {wg},".format(ids=ids, wg=wg[ids]),
              "wb of {ids} is {wb},".format(ids=ids, wb=wb[ids])
              )

        # refresh every 5 seconds
        # print("\n")
        # time.sleep(5)

        # scenario files
        # interval = 60 * 60  # every 60s
        # command = createJson()
        # run(interval, command)

        # States  pvc_charge_power[ids], for house E002
        if ids == "E002":
            pv_e002 = np.array([pvc_charge_power["E002"]])
            load_e002 = np.array([ups_output_power["E002"]])
            p2_e002 = np.array([p2["E002"]])
            rsoc_e002 = np.array([rsoc["E002"]])

            x_e002 = np.concatenate([pv_e002, load_e002, p2_e002, rsoc_e002], axis=-1)
            print(x_e002)

        state_size = (4,)
        action_feature = 3  # batteryStatus, request, accept
        learning_rate = 0.01

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

        # battery = BatteryEnv(action_size=action_size)
        # how the battery changes: from APIS
        # action: scenario generation variables (request, accept, etc..)
        # action refresh to create new scenarios

        memory = Memory(memory_size)

        np.random.seed(42)

    time.sleep(5)


############################
env = House()
env.seed(21)

MEMORY_SIZE = 10000

sess = tf.Session()
with tf.variable_scope('natural_DQN'):
    RL_natural = DQNPrioritizedReplay(
        n_actions=8, n_features=5, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=False, output_graph=True,
    )


with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=8, n_features=5, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
    )
sess.run(tf.global_variables_initializer())


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


def train(RL):
    print("training start")
    total_steps = 0
    steps = []
    episodes = []
    # EPI = 15

    # house_id = input('input the house id: ')

    for i_episode in range(24):

        # TODO: agent needs to get value from the env, not given
        # reset with the env?
        observation = env.reset()
        start_time = time.time()

        while True:  # not gl.sema:

            actions = RL.choose_actions(observation)
            action_request = [actions[0], actions[2]]
            action_accept = [actions[1]]

            agent.CreateSce2(action_request, action_accept)

            # house_id = input('input the house id: ')
            observation_, reward, info = env.step1(action_request, action_accept, house_id)

            actions_space = np.linspace(0.2, 0.9, 8).tolist()
            print("Scenario file updated with act_req {}, {} and act_acc {}".format(actions_space[action_request[0]],
                                                                                  actions_space[action_request[1]],
                                                                                  actions_space[action_accept[0]]))

            # change the time step
            time.sleep(60)

            # if time.sleep(5):  # done:
            #     reward = p2_e001

            RL.store_transition(observation, actions, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            # if time.sleep(5):  # done:
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
