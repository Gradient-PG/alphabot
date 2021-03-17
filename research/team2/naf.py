import numpy as np
import gym
import gym_line_follower  # to register environment
from gym_line_follower.envs import LineFollowerEnv

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


# Returns models necessary for the NAF agent
def build_models(env, window_length):
    """
    Q-function can be decomposed into advantage and value functions, which the NAF agent uses
    The necessary models for the NAF agent are:
    V_model which is the state value function approximation
    L_model which is the lower-triangular matrix approximation (on  which advantage function is based on)
    mu_model which outputs actions, that maximize the Q function
    """
    nb_actions = env.action_space.shape[0]

    V_layer_size = 128
    V_model = Sequential()
    V_model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    V_model.add(Dense(V_layer_size))
    V_model.add(Activation("relu"))
    V_model.add(Dense(V_layer_size))
    V_model.add(Activation("relu"))
    V_model.add(Dense(V_layer_size))
    V_model.add(Activation("relu"))
    V_model.add(Dense(1))
    V_model.add(Activation("linear"))

    mu_layer_size = 128
    mu_model = Sequential()
    mu_model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    mu_model.add(Dense(mu_layer_size))
    mu_model.add(Activation("relu"))
    mu_model.add(Dense(mu_layer_size))
    mu_model.add(Activation("relu"))
    mu_model.add(Dense(mu_layer_size))
    mu_model.add(Activation("relu"))
    mu_model.add(Dense(nb_actions))
    mu_model.add(Activation("linear"))

    l_layer_size = 256
    action_input = Input(shape=(nb_actions,), name="action_input")
    observation_input = Input(shape=(window_length,) + env.observation_space.shape, name="observation_input")
    x = Concatenate()([action_input, Flatten()(observation_input)])
    x = Dense(l_layer_size)(x)
    x = Activation("relu")(x)
    x = Dense(l_layer_size)(x)
    x = Activation("relu")(x)
    x = Dense(l_layer_size)(x)
    x = Activation("relu")(x)
    x = Dense((nb_actions * nb_actions + nb_actions) // 2)(x)
    x = Activation("linear")(x)
    L_model = Model(inputs=[action_input, observation_input], outputs=x)

    return V_model, mu_model, L_model


# Creating environment
env = LineFollowerEnv(gui=False)
nb_actions = env.action_space.shape[0]

window_length = 3  # Amount of previous states taken into account
memory = SequentialMemory(limit=100000, window_length=window_length)  # Memory for experience replay
random_process = OrnsteinUhlenbeckProcess(
    theta=0.15, mu=0.0, sigma=0.3, size=nb_actions
)  # Process used for exploration

# Creating models
V_model, mu_model, L_model = build_models(env, window_length)

# Bulding the agent
agent = NAFAgent(
    nb_actions=nb_actions,
    V_model=V_model,
    L_model=L_model,
    mu_model=mu_model,
    memory=memory,
    nb_steps_warmup=100,
    random_process=random_process,
    gamma=0.99,
    target_model_update=1e-3,
)
agent.compile(Adam(lr=0.001, clipnorm=1.0), metrics=["mae"])

# Training
agent.fit(env, nb_steps=100000, visualize=False, verbose=0)

# Saving the weights
agent.save_weights("cdqn_NAF_weights.h5f", overwrite=True)
env = gym.make("LineFollower-v0")

# Evaluating
agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=200)
