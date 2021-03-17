import gym_line_follower
import gym
import logging
import numpy as np
import random
import math

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    env = gym.make("LineFollower-v0")
    env.reset()

    # Set training parameters
    number_of_episodes = 1000
    time_to_close_loop = 2000
    gamma = 0.9
    alpha = 0.3
    start_epsilon = 1
    min_epsilon = 0.05

    actions = [(0, 0), (1, 0), (1, 0.25), (1, 0.5), (1, 0.75), (1, 1), (0, 1), (0.25, 1), (0.5, 1), (0.75, 1), (1, 1)]
    states_x = [u / 100 for u in range(0, 31, 1)]
    states_y = [u / 100 for u in range(-20, 21, 1)]

    logger.info(f"Action space: {actions}")
    logger.info(f"X space: {states_x}")
    logger.info(f"Y space: {states_y}")

    try:
        q_table = np.load("data.npy")
        epsilon = 0.1
        load = True
    except OSError:
        q_table = np.zeros((len(actions), len(states_x), len(states_y)))
        epsilon = 1.0
        load = False

    logger.info(f"Q table: {q_table}")

    for episode in range(number_of_episodes):
        env.reset()
        episode_ended = False
        timestep = 0
        state_x, state_y = 0.0, 0.0
        reward_all = 0
        logger.info(f"episode number: {episode}")
        if not load:
            epsilon -= (start_epsilon - min_epsilon) / number_of_episodes

        while not episode_ended:
            timestep += 1
            if random.uniform(0, 1) < epsilon:
                action_index = random.randrange(len(actions))
                # logger.info(f"random")
            else:
                action_index = q_table.argmax(axis=0)[states_x.index(state_x)][states_y.index(state_y)]

            logger.info(f"Action: {action_index}, move: {actions[action_index]}")
            obsv, reward, loop_closed, info = env.step(actions[action_index])
            next_x, next_y = round(obsv[0] * 100) / 100, round(obsv[1] * 100) / 100
            q_table[action_index][states_x.index(state_x)][states_y.index(state_y)] = q_table[action_index][
                states_x.index(state_x)
            ][states_y.index(state_y)] + alpha * (
                reward
                + gamma * q_table.argmax(axis=0)[states_x.index(next_x)][states_y.index(next_y)]
                - q_table[action_index][states_x.index(state_x)][states_y.index(state_y)]
            )
            state_x, state_y = next_x, next_y
            reward_all += reward
            episode_ended = loop_closed or timestep >= time_to_close_loop
            # logger.info(f"reward: {reward}")
            # logger.info(f"info :{info}")
            if episode_ended:
                logger.info(f"reward : {reward_all}")

    logger.info(f"Q table: {q_table}")
    np.save("data.npy", q_table)
