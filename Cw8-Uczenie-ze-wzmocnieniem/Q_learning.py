# Q-learning algorithm for 'OpenAI gym' environment with discrete states and actions
# author: Radosław Pietkun


import numpy as np
from math import exp
import random


def train_model(env, Q, num_episodes, num_timesteps, bt, discount, temperature):
    success_count = 0
    for e in range(num_episodes):
        observation = env.reset()
        for t in range(num_timesteps):
            action = choose_action(Q, observation, temperature)
            new_observation, reward, done, info = env.step(action)

            if reward > 0:  success_count += 1  # reached goal (reward > 0 only if goal reached)

            # update 1 element in Q matrix
            if not done:
                Q[observation, action] = (1 - bt) * Q[observation, action] + bt * (
                        reward + discount * max(Q[new_observation, :]))
                observation = new_observation
            else:
                Q[observation, action] = (1 - bt) * Q[observation, action] + bt * reward
                break  # end of current episode

    print("Success rate (during training): {}/{}".format(success_count, num_episodes))
    return Q


def test_model(env, Q, num_attempts, num_timesteps, temperature):
    success_count = 0
    for i in range(num_attempts):
        observation = env.reset()
        for t in range(num_timesteps):
            # env.render()
            action = choose_action(Q, observation, temperature, testing_mode=True)
            observation, reward, done, info = env.step(action)
            if reward > 0: success_count += 1  # reached goal
            if done: break  # end of current attempt
    return success_count / num_attempts


def choose_action(Q, observation, temperature, testing_mode=False):
    P = np.zeros(Q.shape[1])  # probabilities for each action
    for i_action in range(Q.shape[1]):
        # Boltzmann strategy
        P[i_action] = exp(Q[observation, i_action] / temperature) / np.sum(np.exp(Q[observation, :] / temperature))

    if not testing_mode:  # draw action by lot (based on probabilities)
        ran_num = random.uniform(0, 1)  # choose randomly 1 point on line segment from 0 to 1
        prob = 0  # variable which sums probabilities, it represents moving across the line segment from 0 to 1
        for j_action in range(P.shape[0]):
            prob += P[j_action]
            if ran_num <= prob:
                action = j_action
                break
        else:
            # sometimes as a result of numerical errors probabilities don't sum exactly to 1
            # when a number very close to 1 (which outranges the sum) is drawn by lot, then the last action is chosen
            action = P.shape[0] - 1

    else:  # in testing mode choose action with highest probability
        best_actions = np.argwhere(P == np.amax(P))  # find all actions that are the best
        action = np.random.choice(best_actions.flatten())  # choose randomly one of them

    return action
