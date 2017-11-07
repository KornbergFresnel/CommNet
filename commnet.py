import numpy as np
import tensorflow as tf
from base import BaseModel


class CommNet(BaseModel):
    def __init__(self, num_leaver=5, num_agents=500, vector_len=128, num_units=10, learning_rate=0.03, batch_size=64,
                 episodes=500):
        super().__init__(num_leaver, num_agents, vector_len, num_units, learning_rate, batch_size, episodes)

        self.base_line = tf.placeholder(tf.float32, shape=(None, 1))
        self.bias = 1e-4

        # ==== create network =====
        with tf.variable_scope("CommNet"):
            self.eval_name = tf.get_variable_scope().name
            self.policy = self._create_network()  # n * 5 * n_actions

            self.reward = self._get_reward()
            self.loss = self._get_loss()
            self.train_op = tf.train.RMSPropOptimizer(self.alpha).minimize(self.loss)
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def _create_network(self):
        # look-up table
        look_up = tf.get_variable("look_up_table", shape=(self.num_agents, self.vector_len),
                                  initializer=tf.random_normal_initializer)

        # CommNet
        h0 = tf.einsum("ijk,kl->ijl", self.input, look_up)
        h1 = self._create_cell("step_first", self.c_meta, h0, h0)
        c1 = self._mean(h1)
        h2 = self._create_cell("step_second", c1, h1, h0)

        dense_weight = tf.get_variable("dense_weight", shape=(self.vector_len, self.n_actions))
        out = tf.einsum("ijk,kl->ijl", h2, dense_weight)

        # soft-max
        soft = tf.nn.softmax(out) + self.bias

        return soft

    def _sample_action(self):
        reshape_policy = tf.reshape(self.policy, shape=(-1, self.n_actions))
        actions = tf.multinomial(tf.log(reshape_policy), num_samples=1)
        one_hot = tf.one_hot(actions, depth=self.n_actions)
        self.one_hot = tf.reshape(one_hot, shape=(-1, self.num_leaver, self.n_actions))

    def _get_reward(self):
        self._sample_action()
        distinct_num = tf.reduce_sum(tf.cast(tf.reduce_sum(self.one_hot, axis=1) > 0, tf.float32), axis=1)
        return distinct_num / self.num_leaver

    def _get_loss(self):
        # loss1: (n,)
        meta = tf.reshape(self.reward - self.base_line, shape=(-1, 1, 1))

        temp1 = -self.one_hot * tf.log(self.policy)
        temp2 = temp1 * tf.tile(meta, [1, 5, 5])
        loss = tf.reduce_mean(tf.reduce_sum(temp2))

        return loss

    def get_reward(self, ids_one_hot):
        # produce
        return self.sess.run(self.reward, feed_dict={
            self.input: ids_one_hot,
            self.mask: self.mask_data,
            self.c_meta: np.zeros((self.batch_size, self.num_leaver, self.vector_len))
        })

    def train(self, ids_one_hot, base_line, itr):
        ids = np.array([np.random.choice(self.num_agents, 1, replace=False)
                        for _ in range(self.batch_size)])
        ids_one_hot = tf.one_hot(ids, self.vector_len)

        _, loss, policy = self.sess.run([self.train_op, self.loss, self.policy], feed_dict={
            self.input: ids_one_hot,
            self.mask: self.mask_data,
            self.c_meta: np.zeros((self.batch_size, self.num_leaver, self.vector_len)),
            self.base_line: base_line
        })

        if (itr + 1) % 100 == 0:
            print("loss of actor: ", loss)


class BaseLine(BaseModel):
    def __init__(self, num_leaver=5, num_agents=500, vector_len=128, num_units=10, learning_rate=0.03, batch_size=64,
                 episodes=500):
        super().__init__(num_leaver, num_agents, vector_len, num_units, learning_rate, batch_size, episodes)

        self.n_actions = 1
        self.eta = 0.003

        self.reward = tf.placeholder(tf.float32, shape=(None, self.num_leaver, self.n_actions))

        # ==== create network =====
        with tf.variable_scope("CommNet"):
            self.eval_name = tf.get_variable_scope().name
            self.baseline = self._create_network()  # n * 5 * n_actions

            # cross entropy: n * 5 * 1
            self.loss = self._get_loss()
            self.train_op = tf.train.RMSPropOptimizer(self.alpha).minimize(self.loss)
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def _create_network(self):
        # look-up table
        look_up = tf.get_variable("look_up_table", shape=(self.num_agents, self.vector_len),
                                  initializer=tf.random_normal_initializer)

        # CommNet-Baseline
        h0 = tf.einsum("ijk,kl->ijl", self.input, look_up)
        h1 = self._create_cell("step_first", self.c_meta, h0, h0)
        c1 = self._mean(h1)
        h2 = self._create_cell("step_second", c1, h1, h0)
        out = tf.layers.dense(h2, units=self.n_actions, use_bias=1e-4, activation=tf.nn.sigmoid())

        return out

    def _get_loss(self):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.baseline - self.reward))) * self.eta
        return loss

    def get_reward(self, ids_one_hot):
        return self.sess.run(self.baseline, feed_dict={
            self.input: ids_one_hot,
            self.mask: self.mask_data,
            self.c_meta: np.zeros((self.batch_size, self.num_leaver, self.vector_len))
        })

    def train(self, ids_one_hot, reward, itr):
        ids = np.array([np.random.choice(self.num_agents, 1, replace=False)
                        for _ in range(self.batch_size)])
        ids_one_hot = tf.one_hot(ids, self.vector_len)

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.input: ids_one_hot,
            self.mask: self.mask_data,
            self.c_meta: np.zeros((self.batch_size, self.num_leaver, self.vector_len)),
            self.reward: reward
        })

        if (itr + 1) % 100 == 0:
            print("loss of critic: ", loss)





