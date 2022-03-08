"""
DQN training, single run, house E001

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
import analyser
import core
import config as conf
import requests, json

import numpy as np

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

# action
i = 0
decay = 0
while i < 10:
    exp_tradeoff = np.random.rand()
    explore_prob = 0.01 + (1.0 - 0.01) * np.exp(-0.001 * decay)
    if explore_prob > exp_tradeoff:
        action = np.random.randint(0, 7)
    else:
        action = 15
    decay += 1
    i += 1



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
    agent.CreateSce(action_request, action_accept)

    print("req_act: ", action_request_space[action_request[0]], action_request_space[action_request[1]],
          "acc_act: ", action_accept_space[action_accept[0]])
    time.sleep(60)  # 5s

############################
env = House()
env.seed(21)

MEMORY_SIZE = 10000

sess = tf.Session()
with tf.variable_scope('natural_DQN'):
    RL_natural = DQNPrioritizedReplay(
        n_actions=8, n_features=5, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=False,
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


# def train(RL):
total_steps = 0
steps = []
episodes = []
for i_episode in range(15):
    observation = env.reset()
    start_time = time.time()
    while True:

        actions = RL_natural.choose_actions(observation)
        print(actions, type(actions))
        action_request = [actions[0], actions[-1]]
        print(action_request)
        action_accept = actions[1]
        print(action_accept)

        observation_, reward, info = env.step1(observation, action_request, action_accept)

        if done:
            reward = p2_e001

        RL_natural.store_transition(observation, actions, reward, observation_)

        if total_steps > MEMORY_SIZE:
            RL_natural.learn()

        if done:
            print('episode ', i_episode, ' finished')
            steps.append(total_steps)
            episodes.append(i_episode)
            break

        observation = observation_
        total_steps += 1

    end_time = time.time()
    print("episode {} - training time: {:.2f}mins".format(i_episode, (end_time - start_time) / 60))
# return np.vstack((episodes, steps))


# his_natural = train(RL_natural)
# his_prio = train(RL_prio)

# compare based on first success
# plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
# plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
# plt.legend(loc='best')
# plt.ylabel('total training time')
# plt.xlabel('episode')
# plt.grid()
# plt.show()


