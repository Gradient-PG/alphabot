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
    number_of_episodes = 10000
    time_to_close_loop = 1000
    gamma = 0.9
    alpha = 0.1
    epsilon = 0.1

    actions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    states_x = np.arange(0, 0.35, 0.05)
    states_y = np.arange(-0.2, 0.25, 0.05)

    logger.info(f"Action space: {actions}")
    logger.info(f"X space: {states_x}")
    logger.info(f"Y space: {states_y}")

    q_table = np.zeros((len(actions), len(states_x), len(states_y)))
    logger.info(f"Q table: {q_table}")

    for episode in range(number_of_episodes):
        env.reset()
        episode_ended = False
        timestep = 0
        state_x, state_y = 0, 0
        while not episode_ended:
            timestep += 1
            if random.uniform(0, 1) < epsilon:
                action_index = random.randrange(len(actions))
            else:
                action_index = q_table.argmax(axis=0)[state_x][state_y]
            logger.info(f"Action: {action_index}")
            obsv, reward, loop_closed, info = env.step(actions[action_index])
            next_x, next_y = round(obsv[0] * 20) / 20, round(obsv[1] * 20) / 20
            q_table[action_index][state_x][state_y] = q_table[action_index][state_x][state_y] + alpha * (
                reward + gamma * q_table.argmax(axis=0)[next_x][next_y] - q_table[action_index][state_x][state_y]
            )
            state_x, state_y = next_x, next_y
            episode_ended = loop_closed or timestep >= time_to_close_loop

    logger.info(f"Q table: {q_table}")
