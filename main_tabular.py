"""
Tabular methods, single run, house E001

created by: Qiong
2021-Dec-21

"""
import time

import numpy as np
# import main
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from RL_learn import DQNPrioritizedReplay
from agent import House

# sess = main.sess
# sess = tf.Session()

# with tf.Session() as sess:
#     check_point_path = 'model/E001/'
#     ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
#     # main.RL_prio.restore(sess, ckpt)

#
"""
2 feasible methods for loading saved model:
1. saver.restore(): write same NN
2. keras: rebuild the layers (NN) (x)
"""

# # load stored trained model
# # TODO: check if model is correctly loaded by viewing the detailed variables
# saver = tf.train.import_meta_graph('model/E001/E001_model_prio.meta')
# with tf.Session() as sess:
#     saver.restore(sess, 'model/E001/E001_model_prio')
#     graph = tf.get_default_graph()
    # print(sess.run(tf.get_default_graph().get_tensor_by_name("DQN_with_prioritized_replay/eval_net/l1/w1/Initializer"
    #                                                          "/random_normal/stddev:0")))  # [1]

    # a1 = graph.get_tensor_by_name("DQN_with_prioritized_replay/eval_net/l2/w2:0")
    # sess.run(a1)

# all graph info in the saved model (name, value, etc.)
# tensors_per_node = tf.get_default_graph().as_graph_def().node
# tensor_names = [tensor.name for tensor in tensors_per_node]

"""
# save the graph to "logs/" folder
# $ tensorboard --logdir=logs
# http://localhost:6006/
tf.summary.FileWriter("logs/", sess.graph)
"""

# model_file = tf.train.latest_checkpoint('model/E001')
#
# with tf.Session() as sess:
#     saver.restore(sess, model_file)

# """
#####################
# Testing
env = House(action_request=[7, 5], action_accept=[6])
env.seed(1)


def test(RL):
    print("House E001, testing start")
    total_steps = 0
    steps = []
    episodes = []
    reward_list = []
    EPI = 1   # #.of iter
    N_DAY = 14

    # house_id = input('input the house id: ')

    for i_episode in range(EPI):
        # 1 EPI: test one month data first (shorten from one year)
        # (1day, 24hrs) 24 min, action updated every hour (1 min)
        print("Episode {} starts".format(i_episode))
        day = 0
        hour = 0
        done = False

        # (when reset) agent needs to get value from the env, not given
        # reset with the env
        observation = env.reset_time(house_id)
        total_reward = 0

        while day < N_DAY:  # True:  # not gl.sema: total_steps <= 24 (one day)

            # start_time = time.time()
            # TODO: choose actions (e-greedy) --> select actions from learned model
            # sess.run() placeholder
            # test: True/false test: no explore
            actions = RL.choose_actions(observation)  # argmax(DQN.model.predict(states))

            action_request = [actions[0], actions[2]]
            action_accept = [actions[1]]

            observation_, reward, info = env.step1_time(action_request, action_accept, house_id)

            # actions_space = np.around(np.linspace(0.3, 0.9, 7).tolist(), 1)
            actions_space = np.linspace(0.2, 0.9, 8).tolist()
            print("House E001, Scenario file updated with act_req {}, {} and act_acc {}".format(actions_space[action_request[0]],
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
            if (total_steps > 24*3) and (total_steps % 2 == 0):
                RL.learn()
            # RL.learn()

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


house_id = "E001"
# """
# load stored trained model
# TODO: check if model is correctly loaded by viewing the detailed variables
saver = tf.train.import_meta_graph('model/E001/E001_model_prio.meta')
with tf.Session() as sess:
    saver.restore(sess, 'model/E001/E001_model_prio')
    # graph = tf.get_default_graph()
    # sess.run(tf.global_variables_initializer())

# with tf.variable_scope('DQN_with_prioritized_replay'):
RL_prio_test = DQNPrioritizedReplay(
    n_actions=8, n_features=8, memory_size=10000,
    e_greedy_increment=0.002, sess=None, prioritized=True, test=True, output_graph=True,
)  # n_features: 6 states

# sess.run()

prio_memory, prio_reward = test(RL_prio_test)