import tensorflow as tf
import numpy as np
from commnet import BaseLine, CommNet


N_AGENTS = 500
VECTOR_LEN = 128
BATCH_SIZE = 64


def train(episode):
    actor = CommNet(num_agents=N_AGENTS, vector_len=VECTOR_LEN, batch_size=BATCH_SIZE)
    critic = BaseLine(num_agents=N_AGENTS, vector_len=VECTOR_LEN, batch_size=BATCH_SIZE)

    ids = np.array([np.random.choice(N_AGENTS, 1, replace=False)
                    for _ in range(BATCH_SIZE)])
    ids_one_hot = tf.one_hot(ids, VECTOR_LEN)

    for i in range(episode):
        reward = actor.get_reward(ids_one_hot)
        baseline = critic.get_reward(ids_one_hot)

        actor.train(ids_one_hot, baseline, i)
        critic.train(ids_one_hot, reward, i)


if __name__ == "__main__":
    train(5000)

