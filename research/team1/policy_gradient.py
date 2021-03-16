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


class PolicyNetwork:
    """Class representing Policy Network, containing two neural networks, one for each motor"""

    def __init__(
        self, input_shape: typing.Tuple, bias_mu: float, bias_sigma: float, learning_rate: float, checkpoint_path: str
    ):

        self.input_shape = input_shape
        self.bias_mu = bias_mu
        self.bias_sigma = bias_sigma
        self.learning_rate = learning_rate
        self.checkpoint_path = checkpoint_path
        self.left_motor_network = self.create_single_network()
        self.right_motor_network = self.create_single_network()
        self.loss = self.create_policy_network_loss()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def create_single_network(self) -> tf.keras.Model:
        """
        Create deep neural network for a single motor

        :return: Neural network model
        """
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        hidden1 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=tf.keras.initializers.he_normal())(
            inputs
        )
        hidden2 = tf.keras.layers.Dense(128, activation="relu", kernel_initializer=tf.keras.initializers.he_normal())(
            hidden1
        )

        mu = tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Constant(self.bias_mu),
        )(hidden2)

        sigma = tf.keras.layers.Dense(
            1,
            activation="softplus",
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Constant(self.bias_sigma),
        )(hidden2)

        model = tf.keras.Model(inputs=inputs, outputs=[mu, sigma])
        return model

    def create_policy_network_loss(self) -> typing.Callable:
        """
        Create custom policy loss

        :return: Callable loss function
        """

        def policy_network_loss(
            state: typing.List[float], action: typing.List[float], reward: float
        ) -> typing.List[tf.Tensor]:
            """
            Return loss value for action taken at state

            :param state: Current state of agent
            :param action: Action performed by agent
            :param reward: Reward for action performed at state
            :return: Losses for left and right motor
            """
            losses = []
            for index, motor_network in enumerate((self.left_motor_network, self.right_motor_network)):
                nn_mu, nn_sigma = motor_network(state)

                pdf_value = (
                    tf.exp(-0.5 * ((action[index] - nn_mu) / nn_sigma) ** 2) * 1 / (nn_sigma * tf.sqrt(2 * np.pi))
                )

                log_probability = tf.math.log(pdf_value)

                loss = -reward * log_probability
                losses.append(loss)

            return losses

        return policy_network_loss

    def apply_gradients(self, losses: typing.List[tf.Tensor], tape: tf.GradientTape) -> None:
        """
        Backpropagate the loss through networks

        :param losses: Losses for current step
        :param tape: Persistent gradient tape that recorded operations for differentiation
        :return:
        """
        for loss, network in zip(losses, (self.left_motor_network, self.right_motor_network)):
            grads = tape.gradient(loss, network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, network.trainable_variables))

    def save_weights(self) -> None:
        """
        Save weights for networks to checkpoint files

        :return: None
        """
        self.left_motor_network.save(self.checkpoint_path + "_left.h5")
        self.right_motor_network.save(self.checkpoint_path + "_right.h5")

    def load_weights(self) -> None:
        """
        Load weights for networks from checkpoint files

        :return: None
        """
        self.left_motor_network = tf.keras.models.load_model(self.checkpoint_path + "_left.h5")
        self.right_motor_network = tf.keras.models.load_model(self.checkpoint_path + "_right.h5")

    def __call__(self, state):
        mu_left, sigma_left = self.left_motor_network(state)
        mu_right, sigma_right = self.right_motor_network(state)
        return (mu_left, sigma_left), (mu_right, sigma_right)


class PolicyGradientAgent:
    """Class representing Policy Gradient agent"""

    def __init__(
        self,
        env: LineFollowerEnv,
        bias_mu: float = 0.0,
        bias_sigma: float = 0.5,
        learning_rate: float = 0.001,
        min_replay_buffer_len: int = 100,
        checkpoint_path: str = "weights/policy_gradient",
        load_from_checkpoint: bool = False,
    ):

        self.env = env
        self.state_space = env.observation_space
        self.action_space = env.action_space

        self.replay_buffer: typing.Deque = deque(maxlen=50000)
        self.min_replay_buffer_len = min_replay_buffer_len

        self.policy_network = PolicyNetwork(
            self.state_space.shape, bias_mu, bias_sigma, learning_rate, checkpoint_path
        )
        self.checkpoint_path = checkpoint_path

        if load_from_checkpoint:
            self.policy_network.load_weights()

    def train(self, episode_num: int = 1000) -> None:
        """
        Train agent to perform in environment

        :param episode_num: Number of training episodes
        :return: None
        """
        for episode in range(episode_num):
            total_reward = 0
            timestep = 0
            finished = False
            state = self.env.reset()

            while not finished:
                state = np.array(state)
                state_reshaped = state.reshape([1, state.shape[0]])

                (mu_left, sigma_left), (mu_right, sigma_right) = self.policy_network(state_reshaped)

                action = self.take_action(mu_left, sigma_left, mu_right, sigma_right)

                log.debug(f"Action taken at step {timestep}: {action}")
                log.debug(f"left motor: ({mu_left}, {sigma_left})\t right motor: ({mu_right}, {sigma_right})")

                new_state, reward, finished, info = self.env.step(action)

                with tf.GradientTape(persistent=True) as tape:
                    losses = self.policy_network.loss(state_reshaped, action, reward)
                    log.info(f"Loss: {losses}")
                    self.policy_network.apply_gradients(losses, tape)

                state = new_state
                total_reward += reward
                timestep += 1

                if episode % 10 == 0:
                    self.policy_network.save_weights()

            log.info(f"Episode {episode} finished!\t Total reward: {total_reward:.2f}\t Timesteps: {timestep}")

    def take_action(
        self, mu_left: float, sigma_left: float, mu_right: float, sigma_right: float
    ) -> typing.List[float]:

        """
        Sample action from normal distributions of left and right motor

        :param mu_left: Mean for left motor
        :param sigma_left: Standard deviation for left motor
        :param mu_right: Mean for right motor
        :param sigma_right: Standard deviation for right motor
        :return: Action if form of list of two floats
        """

        action_left_motor = tf.random.normal([1], mean=mu_left, stddev=sigma_left).numpy().flatten()[0]
        action_right_motor = tf.random.normal([1], mean=mu_right, stddev=sigma_right).numpy().flatten()[0]
        return [action_left_motor, action_right_motor]

    def test(self, test_env: LineFollowerEnv) -> float:
        """
        Check how the agent performs in test environment

        :param test_env: Test environment for agent
        :return: Total reward
        """
        self.policy_network.load_weights()
        total_reward = 0
        timestep = 0
        finished = False
        state = test_env.reset()

        while not finished:
            state = np.array(state)
            state_reshaped = state.reshape([1, state.shape[0]])

            (mu_left, sigma_left), (mu_right, sigma_right) = self.policy_network(state_reshaped)

            action = self.take_action(mu_left, sigma_left, mu_right, sigma_right)

            log.info(f"Action taken at step {timestep}: {action}")
            log.info(f"left motor: ({mu_left}, {sigma_left})\t right motor: ({mu_right}, {sigma_right})")

            new_state, reward, finished, info = test_env.step(action)
            state = new_state
            total_reward += reward
            timestep += 1

        log.info(f"Test finished!\t Total reward: {total_reward:.2f}\t Timesteps: {timestep}")
        return total_reward


if __name__ == "__main__":
    train_env = LineFollowerEnv(gui=False, max_time=30)

    log.info(f"Observation space: {train_env.observation_space}")
    log.info(f"Action space length: {train_env.action_space}")

    agent = PolicyGradientAgent(train_env)
    agent.train()

    test_env = LineFollowerEnv(gui=True, max_time=30)
    agent.test(test_env)
