import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib


class CommNet:
    def __init__(self, num_leaver=5, num_agents=500, num_units=10, step=2, learning_rate=0.003):
        self.num_leaver = 5
        self.num_agents = 500
        self.alpha = learning_rate
        self.vector_len = 128
        self.num_units = num_units
        self.look_up = np.random.random(size=(self.num_agents, self.vector_len))

        self.input = tf.placeholder(tf.float32, (None, 128))

        with tf.variable_scope("eval_net"):
            self.eval_name = tf.get_variable_scope().name
            self.w = tf.get_variable("w0", tf.random_normal_initializer())
            h1 = tf.matmul(self.input, self.w1)
            self.out = self._create_network(h1)

    def _rnn_cell(self, h, c):
        H = tf.get_variable("rnn_w_h", shape=(self.state_size, self.state_size))
        C = tf.get_variable("rnn_w_c", shape=(self.state_size, self.state_size))

        temp_h = tf.matmul(h, H)
        temp_c = tf.matmul(c, C)

        return tf.nn.tanh(temp_h + temp_c)

    def train_rl(self):
        

    def train_super(self):
        pass
