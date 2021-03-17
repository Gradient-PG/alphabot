import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from gym_line_follower.envs import LineFollowerEnv
import gym
import logging
import random
import numpy as np
import tensorflow as tf
import typing
from collections import deque
from itertools import product

logging.basicConfig(format="%(levelname)s line %(lineno)d: %(message)s", level=logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.FATAL)
log = logging.getLogger(__name__)


class DQAgent:
    """Class representing Deep Q-Learning agent"""

    def __init__(
        self,
        train_env: LineFollowerEnv,
        q_network_learning_rate: float = 0.001,
        min_replay_buffer_len: int = 100,
        checkpoint_path: str = "weights/dqn.ckpt",
        load_from_checkpoint: bool = False,
    ) -> None:
        """
        Initialize agent

        :param train_env: Environment for training
        :param q_network_learning_rate: Learning rate for neural network predicting Q
        :param min_replay_buffer_len: Minimum length of replay buffer to sample from it
        :param checkpoint_path: Path to save/restore model checkpoints
        :param load_from_checkpoint: Load weights from checkpoint during initialization of agent
        """

        self.train_env = train_env
        self.state_space = train_env.observation_space
        self.action_space = train_env.action_space

        self.q_network_learning_rate = q_network_learning_rate
        self.replay_buffer: typing.Deque = deque(maxlen=50000)
        self.min_replay_buffer_len = min_replay_buffer_len

        self.q_network = self.create_q_network()
        self.checkpoint_path = checkpoint_path

        if load_from_checkpoint:
            self.q_network.load_weights(self.checkpoint_path)

    def create_q_network(self) -> tf.keras.Sequential:
        """
        Create neural network predicting Q values

        :return: Keras model of neural network
        """
        state_shape = self.state_space.shape
        action_shape = len(self.action_space)
        init = tf.keras.initializers.HeUniform()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, input_shape=state_shape, activation="relu", kernel_initializer=init))
        model.add(tf.keras.layers.Dense(128, activation="relu", kernel_initializer=init))
        model.add(tf.keras.layers.Dense(action_shape, activation="relu", kernel_initializer=init))
        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.q_network_learning_rate),
            metrics=["accuracy"],
        )
        return model

    def _fit_network(self, q_value_learning_rate: float, discount_factor: float, batch_size: int) -> None:
        """
        Change network weights based on observations from replay buffer

        :param q_value_learning_rate: Learning rate used in Q-learning algorithm
        :param discount_factor: Discount factor for rewards
        :param batch_size: Size of batch sampled from replay buffer
        :return: None
        """
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path, save_weights_only=True, verbose=False
        )

        if len(self.replay_buffer) < self.min_replay_buffer_len:
            return

        mini_batch = random.sample(self.replay_buffer, batch_size)
        current_states = np.array([observation[0] for observation in mini_batch])
        current_q_list = self.q_network.predict(current_states)
        new_current_states = np.array([observation[3] for observation in mini_batch])
        future_q_list = self.q_network.predict(new_current_states)

        x, y = [], []
        for index, (observation, action, reward, new_observation, finished) in enumerate(mini_batch):
            if not finished:
                max_future_q = reward + discount_factor * np.max(future_q_list[index])
            else:
                max_future_q = reward

            current_qs = current_q_list[index]
            log.debug(f"Current q list: {current_q_list}")
            log.debug(f"Current qs: {current_qs}")
            log.debug(f"Current action: {action}")
            action_index = self.action_space.index(action)
            log.debug(f"Max future q: {max_future_q}")
            log.debug(f"Max future reward: {np.max(future_q_list[index])}")
            current_qs[action_index] = (1 - q_value_learning_rate) * current_qs[
                action_index
            ] + q_value_learning_rate * max_future_q

            x.append(observation)
            y.append(current_qs)

        self.q_network.save_weights(self.checkpoint_path.format(epoch=0))
        log.debug(f"x: {x}")
        log.debug(f"y: {y}")
        self.q_network.fit(
            np.array(x), np.array(y), batch_size=batch_size, verbose=False, shuffle=True, callbacks=[cp_callback]
        )

    def train_agent(
        self,
        episode_num: int = 400,
        epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.01,
        q_value_learning_rate: float = 0.7,
        discount_factor: float = 0.5,
        batch_size: int = 64,
    ) -> None:
        """
        Train agent to perform in environment

        :param episode_num: Number of training episodes
        :param epsilon: Probability of choosing a random action instead of predicting from Q-network
        :param min_epsilon: Minimum value of epsilon
        :param epsilon_decay: Rate of epsilon decay per episode
        :param q_value_learning_rate: Learning rate used in Q-learning algorithm
        :param discount_factor: Discount factor for rewards
        :param batch_size: Size of batch sampled from replay buffer
        :return: None
        """
        for episode in range(episode_num):
            train_env = LineFollowerEnv(gui=False, max_time=20)
            train_env.action_space = [(0.5, 0), (0, 0.5), (1, 0), (0, 1), (1, 1)]

            total_reward = 0
            finished = False
            observation = train_env.reset()
            timestep = 0

            while not finished:
                timestep += 1
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(self.action_space)
                    log.debug(f"Sampled action: {action}")
                else:
                    observation = np.array(observation)
                    observation_reshaped = observation.reshape([1, observation.shape[0]])
                    predicted = self.q_network.predict(observation_reshaped).flatten()
                    action = self.action_space[np.argmax(predicted)]
                    log.debug(f"Predictions: {predicted}\tbest action: {action}")

                new_observation, reward, finished, info = train_env.step(action)
                self.replay_buffer.append([observation, action, reward, new_observation, finished])
                log.debug(f"Episode {episode}, timestep {timestep}, reward: {reward}")

                self._fit_network(q_value_learning_rate, discount_factor, batch_size)
                observation = new_observation
                total_reward += reward

                if finished:
                    log.info(f"Total reward: {total_reward} after n steps = {episode}, RAP: {epsilon}")

            epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-epsilon_decay * episode)

    def test_agent(self, test_env: LineFollowerEnv) -> float:
        """
        Check how the agent performs in test environment

        :param test_env: Test environment for agent
        :return: Total reward
        """
        self.q_network.load_weights(self.checkpoint_path)
        observation = test_env.reset()
        total_reward = 0
        finished = False
        log.info("Testing agent")
        while not finished:
            observation = np.array(observation)
            observation_reshaped = observation.reshape([1, observation.shape[0]])
            predicted = self.q_network.predict(observation_reshaped).flatten()
            action = test_env.action_space[np.argmax(predicted)]
            log.debug(f"Predictions: {predicted},\t\taction: {action}")
            new_observation, reward, finished, info = test_env.step(action)
            total_reward += reward
            observation = new_observation

        return total_reward


if __name__ == "__main__":
    train_env = LineFollowerEnv(gui=False)
    left_motor_speeds = np.linspace(0.0, 1.0, 7)
    right_motor_speeds = np.linspace(0.0, 1.0, 7)
    train_env.action_space = [action for action in product(left_motor_speeds, right_motor_speeds)]

    log.info(f"Observation space: {train_env.observation_space}")
    log.info(f"Action space length: {len(train_env.action_space)}")

    agent = DQAgent(train_env)
    # agent.train_agent()

    test_env = LineFollowerEnv(gui=True, max_time=30)
    left_motor_speeds = np.linspace(0.0, 1.0, 7)
    right_motor_speeds = np.linspace(0.0, 1.0, 7)
    test_env.action_space = [action for action in product(left_motor_speeds, right_motor_speeds)]
    total_reward = agent.test_agent(test_env)
    log.info(f"Test total reward: {total_reward}")
