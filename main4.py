#!/usr/bin/env python
"""
DQN training, single run, house E004

created by: Qiong

"""
import pickle
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
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
from RL_learn import Memory, DQNPrioritizedReplay
from agent import APIS, House


############################


MEMORY_SIZE = 10000  # 10000

sess = tf.Session()

# with tf.variable_scope('natural_DQN'):
#     RL_natural = DQNPrioritizedReplay(
#         n_actions=8, n_features=8, memory_size=MEMORY_SIZE,
#         e_greedy_increment=0.008, sess=sess, prioritized=False, output_graph=True,
#     )


with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=8, n_features=8, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.002, sess=sess, prioritized=True, test=False, output_graph=True,
    )
sess.run(tf.global_variables_initializer())

# env = House(action_request=[7, 5], action_accept=[6])
# env.seed(1)


def train(RL):
    print("House E004, training start")
    total_steps = 0
    steps = []
    episodes = []
    reward_list = []
    EPI = 3
    N_RUN = 3
    N_DAY = 30

    for i_run in range(N_RUN):

        env = House(action_request=[7, 5], action_accept=[6])
        env.seed(1)
        print("********Run {} starts********".format(i_run+1))
        total_reward = 0

        # reset with the env
        observation = env.reset_time(house_id)

        for i_episode in range(EPI):

            print("Episode {} starts".format(i_episode))
            day = 0
            hour = 0
            done = False

            # # reset with the env
            # observation = env.reset_time(house_id)
            # start_time = time.time()
            # total_reward = 0

            while day < N_DAY:  # True:  # not gl.sema:

                actions = RL.choose_actions(observation)
                action_request = [actions[0], actions[2]]
                action_accept = [actions[1]]

                # agent.CreateSce4(action_request, action_accept)

                # house_id = input('input the house id: ')
                observation_, reward, info = env.step4_time(action_request, action_accept, house_id)

                actions_space = np.linspace(0.2, 0.9, 8).tolist()
                # actions_space = np.around(np.linspace(0.3, 0.9, 7).tolist(), 1)
                print("House E004, Scenario file updated with act_req {}, {} and act_acc {}".format(actions_space[action_request[0]],
                                                                                      actions_space[action_request[1]],
                                                                                      actions_space[action_accept[0]]))

                RL.store_transition(observation, actions, reward, observation_)

                # reward_list.append(reward)
                total_reward += reward

                # if total_steps > MEMORY_SIZE:
                if (total_steps > 24*3) and (total_steps % 2 == 0):
                    RL.learn()

                if hour < 24/3:  # 24 - 1:#(total_steps > 0) and (total_steps % 24 == 0):  # one day
                    hour += 1
                    observation = observation_
                    total_steps += 1
                    print("total_steps = ", total_steps)
                    time.sleep(0)  # update every hour
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

            # Track rewards
            reward_list.append(total_reward)
            # end_time = time.time()
            # print("episode {} - training time: {:.2f}mins".format(i_episode, (end_time - start_time) / 60 * gl.acc))

        time.sleep(10)
        print("*******save model start for RUN {} *******".format(i_run+1))
        # save model after each iter is over (set checkpoint? --for certain numbers of steps)
        saver = tf.train.Saver()
        saver.save(RL.sess, 'model/E004/E004_model_prio')
        print('Model Trained and Saved')

    # return np.vstack((episodes, steps)), RL.memory
    return RL.memory, reward_list


house_id = "E004"  # input('input the house id: ')
# his_natural, natural_memory = train(RL_natural)
# natural_memory, natural_reward = train(RL_natural)
# #  save memo to json file
# with open("saved/natural_memo_e004_May_iter1_time.data", "wb") as fp:
#     pickle.dump(natural_memory, fp)
# #  save reward to json file
# with open("saved/natural_reward_e004_May_iter1_time.data", "wb") as fp:
#     pickle.dump(natural_reward, fp)

# his_prio, prio_memory = train(RL_prio)
prio_memory, prio_reward = train(RL_prio)
# prio_memory_store = [prio_memory.tree.data[i][9] for i in range(24*55)]  # reward(p2)
# save memo to json file
with open("saved/prio_memo_e004_May_iter5_train_time.data", "wb") as fp:
    pickle.dump(prio_memory, fp)
# save reward to json file
with open("saved/prio_reward_e004_May_iter5_train_time.data", "wb") as fp:
    pickle.dump(prio_reward, fp)

# compare based on first success
# plt.title("E004")

# plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='g', label='natural DQN')
# plt.plot(natural_memory[:24, 8], 'b', label='natural DQN p2')

# plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='b', label='DQN with prioritized replay')
# plt.plot(prio_memory_store, 'g', label='DQN with prioritized replay')
# plt.legend(loc='best')
# plt.ylabel('reward (p2)')
# plt.xlabel('episode (hour)')
# plt.grid()
# plt.show()
