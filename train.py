import tensorflow as tf
import numpy as np
from commnet import BaseLine, CommNet


N_AGENTS = 50
VECTOR_LEN = 128
BATCH_SIZE = 64
LEVER = 5


def train(episode):
    actor = CommNet(num_agents=N_AGENTS, vector_len=VECTOR_LEN, batch_size=BATCH_SIZE)
    critic = BaseLine(num_agents=N_AGENTS, vector_len=VECTOR_LEN, batch_size=BATCH_SIZE)

    id_1 = np.array([np.random.choice(N_AGENTS, LEVER, replace=False)])

    for i in range(episode):
        ids = np.array([np.random.choice(N_AGENTS, LEVER, replace=False)
                        for _ in range(BATCH_SIZE)])

        reward = actor.get_reward(ids)
        baseline = critic.get_reward(ids)

        actor.train(ids, baseline, i)
        critic.train(ids, reward, i)

        if (i + 1) % 10 == 0:
            print(actor.pred(id_1))


if __name__ == "__main__":
    train(500)

