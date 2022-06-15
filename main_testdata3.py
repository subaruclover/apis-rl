#!/usr/bin/env python
"""
DQN testing, single run, house E003
Load the trained model from model/houseID file

testing data (each has 2 weeks, 14 DAYS, 2019):
i) Feb 8-21
ii) Jun 14-27
iii) Jul 21 - Aug 3

created by: Qiong

"""
import pickle
import time
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from RL_learn import DQNPrioritizedReplay
from agent import House

env = House(action_request=[7, 5], action_accept=[6])
env.seed(1)


# Testing function
def test(RL):
    print("House E003, testing start")
    total_steps = 0
    steps = []
    episodes = []
    reward_list = []
    EPI = 1  # #.of iter
    N_DAY = 14

    # house_id = input('input the house id: ')

    for i_episode in range(EPI):

        print("Episode {} starts".format(i_episode))
        day = 0
        hour = 0
        done = False

        # (when reset) agent needs to get value from the env, not given
        # reset with the env
        observation = env.reset_time(house_id)
        total_reward = 0

        while day < N_DAY:

            # TODO: choose actions (e-greedy) --> select actions from learned model
            # sess.run() placeholder
            # test: True/false, while test: no explore
            actions = RL.choose_actions(observation)  # argmax(DQN.model.predict(states))

            action_request = [actions[0], actions[2]]
            action_accept = [actions[1]]

            observation_, reward, info = env.step3_time(action_request, action_accept, house_id)

            # actions_space = np.around(np.linspace(0.3, 0.9, 7).tolist(), 1)
            actions_space = np.linspace(0.2, 0.9, 8).tolist()
            print("House E003, Scenario file updated with act_req {}, {} and act_acc {}".format(
                actions_space[action_request[0]],
                actions_space[action_request[1]],
                actions_space[action_accept[0]]))
            # Store the experience in memory
            RL.store_transition(observation, actions, reward, observation_)

            # reward_list.append(reward)
            total_reward += reward
            # start learn after 100 steps and the frequency of learning
            # accumulate some memory before start learning
            # if (total_steps > 24 * 3) and (total_steps % 2 == 0):
            RL.learn()

            if hour < 24 / 3:  # 24 - 1:#(total_steps > 0) and (total_steps % 24 == 0):  # one day
                hour += 1
                observation = observation_
                total_steps += 1
                print("total_steps = ", total_steps)

                time.sleep(0.01)  # update every 3 hours
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
            # print("total_steps = ", total_steps)
        # Track rewards
        reward_list.append(total_reward)
        # end_time = time.time()
        # print("episode {} - training time: {:.2f}mins".format(i_episode, (end_time - start_time) / 60 * gl.acc))

    print('Testing end')
    # return np.vstack((episodes, steps)), RL.memory
    return RL.memory, reward_list


house_id = "E003"
# """
# load stored trained model
# TODO: check if model is correctly loaded by viewing the detailed variables
saver = tf.train.import_meta_graph('model/E003/E003_model_prio.meta')
with tf.Session() as sess:
    saver.restore(sess, 'model/E003/E003_model_prio')
    # graph = tf.get_default_graph()
    # sess.run(tf.global_variables_initializer())

# with tf.variable_scope('DQN_with_prioritized_replay'):
RL_prio_test = DQNPrioritizedReplay(
    n_actions=8, n_features=8, memory_size=10000,
    e_greedy_increment=0.003, sess=None, prioritized=True, test=True, output_graph=True,
)  # n_features: 6 states

# sess.run()

prio_memory, prio_reward = test(RL_prio_test)

# save memo to json file
with open("saved/prio_memo_e003_Feb_test.data", "wb") as fp:
    pickle.dump(prio_memory, fp)
# save reward to json file
with open("saved/prio_reward_e003_Feb_test.data", "wb") as fp:
    pickle.dump(prio_reward, fp)
