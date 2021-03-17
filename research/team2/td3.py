import gym
import gym_line_follower  # to register environment
from gym_line_follower.envs import LineFollowerEnv

import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


# Returns agent for the given enviroment
def create_agent(env):
    # The noise objects for TD3
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # Create the agent
    return TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)


# Trains agent and saves it's weights
def train_agent(agent):
    agent.learn(total_timesteps=1000, log_interval=10)
    agent.save("td3_line_follower")


# Tests given agent in given enviroment
def test_agent(env, agent):
    obs = env.reset()
    while True:
        action, _states = agent.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()


if __name__ == "__main__":
    env = LineFollowerEnv(gui=False)

    agent = create_agent(env)
    train_agent(agent)

    env = gym.make("LineFollower-v0")

    test_agent(env, agent)
