import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from gym_line_follower.envs import LineFollowerEnv
import gym
import logging
import numpy as np
import tensorflow as tf
import typing
from collections import deque
from itertools import product

logging.basicConfig(format="%(levelname)s line %(lineno)d: %(message)s", level=logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.FATAL)
log = logging.getLogger(__name__)


class PolicyGradientAgent:
    """Class representing Policy Gradient agent"""

    def __init__(self,
                 env: LineFollowerEnv,
                 bias_mu: float = 0.0,
                 bias_sigma: float = 0.5,
                 min_replay_buffer_len: int = 100,
                 checkpoint_path: str = "weights/policy_gradient.h5",
                 load_from_checkpoint: bool = False
                 ):
        self.env = env
        self.bias_mu = bias_mu
        self.bias_sigma = bias_sigma
        self.state_space = env.observation_space
        self.action_space = env.action_space

        self.replay_buffer: typing.Deque = deque(maxlen=50000)
        self.min_replay_buffer_len = min_replay_buffer_len

        self.policy_network = self.create_policy_network()
        self.policy_loss = self.create_policy_network_loss()
        self.checkpoint_path = checkpoint_path

        if load_from_checkpoint:
            self.q_network.load_weights(self.checkpoint_path)

    def create_policy_network(self) -> tf.keras.Model:
        """
        Create policy network

        :return: Policy network
        """
        inputs = tf.keras.layers.Input(shape=(16,))
        hidden1 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=tf.keras.initializers.he_normal())(inputs)
        hidden2 = tf.keras.layers.Dense(128, activation="relu", kernel_initializer=tf.keras.initializers.he_normal())(hidden1)

        mu = tf.keras.layers.Dense(1, activation="linear",
                                   kernel_initializer=tf.keras.initializers.Zeros(),
                                   bias_initializer=tf.keras.initializers.Constant(self.bias_mu))(hidden2)

        sigma = tf.keras.layers.Dense(1, activation="softplus",
                                      kernel_initializer=tf.keras.initializers.Zeros(),
                                      bias_initializer=tf.keras.initializers.Constant(self.bias_sigma))(hidden2)

        policy_network = tf.keras.Model(inputs=inputs, outputs=[mu, sigma])
        return policy_network

    def create_policy_network_loss(self) -> float:
        """
        Create custom policy loss

        :return: Loss
        """
        def policy_network_loss(state, action, reward):
            nn_mu, nn_sigma = self.policy_network(state)

            # Obtain pdf of Gaussian distribution
            pdf_value = tf.exp(-0.5 * ((action - nn_mu) / nn_sigma) ** 2) * 1 / (nn_sigma * tf.sqrt(2 * np.pi))

            # Compute log probability
            log_probability = tf.math.log(pdf_value + 1e-5)

            # Compute weighted loss
            loss = - reward * log_probability
            return loss

        return policy_network_loss

    def train(self,
              episode_num: int = 10000,
              learning_rate: float = 0.001
              ):

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for episode in range(episode_num):
            total_reward = 0
            timestep = 0
            finished = False
            state = self.env.reset()

            while not finished:
                state = np.array(state)
                state_reshaped = state.reshape([1, state.shape[0]])

                # Obtain mu and sigma from network
                mu, sigma = self.policy_network(state_reshaped)

                # Draw action from normal distribution
                action = tf.random.normal([2], mean=mu, stddev=sigma)
                action = tf.squeeze(action)
                log.debug(f"Action taken at step {timestep}: {action}")

                # Compute reward
                new_state, reward, finished, info = self.env.step(action)

                # Update network weights
                with tf.GradientTape() as tape:
                    # Compute Gaussian loss
                    loss_value = self.policy_loss(state_reshaped, action, reward)

                    # Compute gradients
                    grads = tape.gradient(loss_value, self.policy_network.trainable_variables)

                    # Apply gradients to update network weights
                    optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))

                state = new_state
                total_reward += reward
                timestep += 1

                if episode % 10 == 0:
                    self.policy_network.save(self.checkpoint_path)

            log.info(f"Episode {episode} finished!\t Total reward: {total_reward:.2f}\t Timesteps: {timestep}")

    def test(self, test_env: LineFollowerEnv) -> float:
        """
        Test agent

        :param test_env:
        :return:
        """
        self.policy_network = tf.keras.models.load_model(self.checkpoint_path)
        total_reward = 0
        timestep = 0
        finished = False
        state = test_env.reset()

        while not finished:
            state = np.array(state)
            state_reshaped = state.reshape([1, state.shape[0]])

            # Obtain mu and sigma from network
            mu, sigma = self.policy_network(state_reshaped)

            # Draw action from normal distribution
            action = tf.random.normal([2], mean=mu, stddev=sigma)
            action = tf.squeeze(action)
            log.info(f"Action taken at step {timestep}: {action}")

            # Compute reward
            new_state, reward, finished, info = test_env.step(action)

            state = new_state
            total_reward += reward
            timestep += 1

        log.info(f"Test finished!\t Total reward: {total_reward:.2f}\t Timesteps: {timestep}")

if __name__ == "__main__":
    train_env = LineFollowerEnv(gui=False, max_time=30)

    log.info(f"Observation space: {train_env.observation_space}")
    log.info(f"Action space length: {train_env.action_space}")

    agent = PolicyGradientAgent(train_env)
    # print(agent.policy_network.summary())
    agent.train()

    #total_reward = agent.test_agent(test_env)
    #log.info(f"Test total reward: {total_reward}")
