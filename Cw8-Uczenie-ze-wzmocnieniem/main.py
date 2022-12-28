# Train and test model with Q-learning algorithm
# author: Radosław Pietkun


import gym
from Q_learning import train_model, test_model
import numpy as np


if __name__ == '__main__':
    num_timesteps = 200  # max number of steps in 1 episode
    num_episodes = int(1e5)  # number of episodes during learning
    num_attempts = int(1e2)  # number of attempts during testing
    bt = 0.1  # step in Q-learning
    discount = 0.99
    temperature = int(1e2)  # param for Boltzmann strategy

    env = gym.make("FrozenLake8x8-v0")
    # env.action_space:  Discrete(4)
    # env.observation_space: Discrete(64)

    Q = np.zeros([env.observation_space.n, env.action_space.n])  # empty value-action matrix

    print("Training model...")
    Q = train_model(env, Q, num_episodes, num_timesteps, bt, discount, temperature)
    print("Model trained")
    file1 = "Q_matrix.txt"
    np.savetxt(file1, Q)
    #Q = np.loadtxt("Q_matrix.txt")

    print("\nTesting model...")
    success_rate = test_model(env, Q, num_attempts, num_timesteps, temperature)
    print("Model tested with success rate: {:.2f}%".format(success_rate*100))

    env.close()
