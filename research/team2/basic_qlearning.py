import numpy as np
import pickle
import random
from tensorflow.keras import layers
import gym
import gym_line_follower  # to register environment


# Generates an index for random action
def random_action(num_actions: int) -> int:
    action = np.random.randint(0, num_actions)
    return action


# Turns given x and y into their index in the state list
def x_y_to_state_idx(x: float, y: float) -> int:
    x_idx = -1
    for i in np.arange(0, 0.3 + x_step, x_step):
        if i > x:
            break
        x_idx += 1

    y_idx = -1
    for i in np.arange(-0.2, 0.2 + y_step, y_step):
        if i > y:
            break
        y_idx += 1
    return states.index((states_x[x_idx], states_y[y_idx]))


# Tranforms observation from enviroment into (x,y) state
def observation_to_state(obs: tuple) -> int:
    x, y = obs[0], obs[1]
    return x_y_to_state_idx(x, y)


def train(
    env: gym_line_follower.envs.line_follower_env.LineFollowerEnv,
    Q: list,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    num_actions: int,
    checkpoint_name: str = "q_table",
) -> None:
    """
    Body of q-learning algorithms
    """

    for i in range(episodes):
        obs = env.reset()
        state = observation_to_state(obs)
        reward_sum = 0
        update_interval = 10
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                """
                Exploration: doing random action
                """
                action_idx = random_action(num_actions)
            else:
                """
                Exploitation: doing the best action
                """
                action_idx = np.argmax(Q[state])

            """
            Performing an action.
            """
            action = actions[action_idx]
            next_obs, reward, done, _ = env.step(action)
            next_state = observation_to_state(next_obs)
            reward_sum += reward
            """
            Updating Q-table.
            """
            old_value = Q[state][action_idx]
            next_max = np.max(Q[next_state])
            Q[state][action_idx] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            state = next_state
        if i % update_interval == 0:
            print("Episode: " + str(i))
            print("Average reward: " + str(reward_sum / update_interval))
            reward_sum = 0

    checkpoint_name += f"_{episodes}"

    with open(checkpoint_name, "wb") as file:
        pickle.dump(Q, file)


if __name__ == "__main__":
    # initialization
    # Creating enviroment
    env = gym.make("LineFollower-v0")
    x_step = 0.025  # Jump between discretized x states
    y_step = 0.025  # Jump between discretized y states
    rotor_step = 0.25  # Jump between discretized rotor values
    # Q-learning parameters
    episodes = 1000
    epsilon = 0.1  # Chance of exploration
    alpha = 0.1  # learning rate
    gamma = 0.6  # Discount factor

    states_rotor = np.arange(0, 1 + rotor_step, rotor_step)  # Possible rotor actions
    # Create a list of tuples, with every possible state
    states_x = np.arange(0, 0.3 + x_step, x_step)  # Possible x states
    states_y = np.arange(-0.2, 0.2 + y_step, y_step)  # Possible y states
    states = [(x, y) for y in states_y for x in states_x]
    # Create a list of tuples, with every possible action
    actions = [
        (rotor_one_state, rotor_two_state) for rotor_one_state in states_rotor for rotor_two_state in states_rotor
    ]
    num_actions = len(actions)  # Number of rotor actions
    num_states = len(states)  # Number of states

    Q = np.zeros((num_states, num_actions))  # Q table
    train(env, Q, episodes, alpha, gamma, epsilon, num_actions)
