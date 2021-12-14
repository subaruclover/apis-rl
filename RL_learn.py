"""
# Deep Q Network
# Battery Model

created by: Qiong
09/28/2021
"""

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam


# Deep Q-Network Model
class DQNNet():

    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self.create_model()

    def create_model(self):
        # state_size = (3, )
        input = Input(shape=self.state_size)

        x = Dense(50, activation="relu", kernel_initializer=glorot_uniform(seed=42))(input)
        x = Dense(200, activation="relu", kernel_initializer=glorot_uniform(seed=42))(x)

        output = Dense(
            self.action_size,
            activation="linear",
            kernel_initializer=glorot_uniform(seed=42),
        )(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model


# Memory Model
# A tree based array containing priority of each experience for fast sampling
class SumTree():
    """
    __init__ - create data array storing experience and a tree based array storing priority
    add - store new experience in data array and update tree with new priority
    update - update tree and propagate the change through the tree
    get_leaf - find the final nodes with a given priority value

    store data with its priority in the tree.
    """

    data_pointer = 0

    def __init__(self, capacity):

        """
        capacity - Number of final nodes containing experience, for all priority values
        data - array containing experience (with pointers to Python objects), for all transitions
        tree - a tree shape array containing priority of each experience

        tree index:
            0       -> storing priority sum
           / \
          1   2
         / \ / \
        3  4 5  6   -> storing priority for transitions

        Array type for storing:
        [0, 1, 2, 3, 4, 5, 6]
        """

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):

        # Start from first leaf node of the most bottom layer
        tree_index = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data  # Update data frame
        self.update(tree_index, priority)  # Update priority

        # Overwrite if exceed memory capacity
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):

        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # Propagate the change through tree
        while (
                tree_index != 0
        ):  # this method is faster than the recursive loop in the reference code
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):

        parent_index = 0

        while True:  # while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1  # this leaf's left and right kids
            right_child_index = left_child_index + 1
            # Downward search, always search for a higher priority node till the last layer
            if left_child_index >= len(self.tree):  # reach the bottom, end search
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        # tree leaf index, priority, experience
        return leaf_index, self.tree[leaf_index], self.data[data_index]


# Memory Model
class Memory():  # stored as (s, a, r, s_) in SumTree

    """

    __init__ - create SumTree memory
    store - assign priority to new experience and store with SumTree.add & SumTree.update
    sample - uniformly sample from the range between 0 and total priority and
             retrieve the leaf index, priority and experience with SumTree.get_leaf
    batch_update - update the priority of experience after training with SumTree.update

    PER_e - Hyperparameter that avoid experiences having 0 probability of being taken
    PER_a - Hyperparameter that allows tradeoff between taking only experience with
            high priority and sampling randomly (0 - pure uniform randomness, 1 -
            select experiences with the highest priority)
    PER_b - Importance-sampling, from initial value increasing to 1, control how much
            IS affect learning

    """

    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4
    PER_b_increment_per_sampling = 0.01
    absolute_error_upper = 1.0  # Clipped abs error

    def __init__(self, capacity):

        self.tree = SumTree(capacity)

    def store(self, experience):

        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0, this experience will never have a chance to be selected
        # So a minimum priority is assigned
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)

    def sample(self, n):

        """
        First, to sample a minibatch of k size, the range [0, priority_total] is
        divided into k ranges. A value is uniformly sampled from each range. Search
        in the sumtree, the experience where priority score correspond to sample
        values are retrieved from. Calculate IS weights for each minibatch element
        """

        b_memory = []
        b_idx = np.empty((n,))
        b_ISWeights = np.empty((n, 1))

        priority_segment = self.tree.tree[0] / n

        self.PER_b = np.min([1.0, self.PER_b + self.PER_b_increment_per_sampling])

        prob_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.tree[0]
        max_weight = (prob_min * n) ** (-self.PER_b)

        for i in range(n):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            prob = priority / self.tree.tree[0]
            b_ISWeights[i, 0] = (prob * n) ** (-self.PER_b) / max_weight
            b_idx[i] = index
            b_memory.append([data])

        return b_idx, b_memory, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):

        # To avoid 0 probability
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# DQN, with/without Prioritized Replay
class DQNPrioritizedReplay:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.005,
                 reward_decay=0.9,
                 e_greedy=0.95,
                 replace_target_iter=500,
                 memory_size=10000,
                 batch_size=256,
                 e_greedy_increment=None,
                 output_graph=False,
                 prioritized=True,
                 sess=None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized    # decide to use prioritize experience replay or not

        self.learn_step_counter = 0

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer, trainable):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names,  trainable=trainable)
                out = tf.matmul(l1, w2) + b2
            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer, True)

        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.math.reduce_sum(tf.math.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.math.reduce_mean(self.ISWeights * tf.math.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.math.reduce_mean(tf.math.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer, False)

    def store_transition(self, s, a, r, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.s_: batch_memory[:, -self.n_features:],
                           self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
