from gym_line_follower.envs import LineFollowerEnv
import gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise


# Train
train_env = LineFollowerEnv(gui=False)

n_actions = train_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", train_env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=3)

model.save("ddpg")
train_env.close()

# Test
# model.load("ddpg")

test_env = LineFollowerEnv(gui=True)
state = test_env.reset()

done = False
while not done:
    action, _states = model.predict(state)
    state, reward, done, info = test_env.step(action)

test_env.close()
