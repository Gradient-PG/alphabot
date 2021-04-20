import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import torch.optim as optim
from collections import deque
import gym
import random
import gym_line_follower
from gym_line_follower.envs import LineFollowerEnv
from typing import List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OUNoise:
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    Based on the rllab implementation.
    class from: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        mu: float = 0.0,
        theta: float = 0.15,
        max_sigma: float = 0.3,
        min_sigma: float = 0.3,
        decay_period: int = 100000,
    ):
        """
        @param action_space: Action space returned by enviroment.
        @param mu: Parameter ( mu >0 ).
        @param theta: Parameter (theta > 0).
        @param max_sigma: Parameter
        @param min_sigma: Parameter
        @param decay_period: Process decay period
        """
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self) -> None:
        """
        Resets the state of noise generator.
        """
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self) -> np.array:
        """
        Changes the state of noise generator.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action: torch.FloatTensor, t: int = 0) -> np.array:
        """
        Returns an action with added noise.
        @param action: Tensor with actions.
        @param t: Time step.
        @return: np.array
        """
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class Actor(nn.Module):
    def __init__(self, states_size, actions_num):
        """
        Actor network class. Last layer has tanh activation in order to output value in range (-1,1).
        @param states_size: state vector size
        @param actions_num: size of action space
        """
        super().__init__()
        hidden_size = 128
        self.fc1 = nn.Linear(states_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, actions_num)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = torch.tanh(self.fc3(y))
        return y


class Critic(nn.Module):
    def __init__(self, states_size, actions_num):
        """
        Crtitc network class. Last layer no activation.
        @param states_size: state vector size
        @param actions_num: size of action space
        """
        super().__init__()
        hidden_size = 128
        self.fc1 = nn.Linear(states_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, actions_num)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


class TD3Agent:
    """
    Implements TD3 algorithm.
    """

    def __init__(
        self,
        states_size: int,
        actions_num: int,
        noise: OUNoise,
        buffer_size: int = 100000,
        learning_rate_critic1: float = 0.0005,
        learning_rate_critic2: float = 0.001,
        learning_rate_actor: float = 0.001,
        policy_delay: int = 2,
        batch_size: int = 32,
        tau: float = 0.01,
        gamma: float = 0.99,
    ) -> None:
        """
        Target network soft update.
        :@param: states_size: size of state vector
        :@param: actions_num: size of action space
        :@param: noise: Noise generator
        :@param: buffer_size: size of replay buffer, agents memory
        :@param: learning_rate_critic1: learning rate for critic1
        :@param: learning_rate_critic2: learning rate for critic2
        :@param: learning_rate_actor: learning rate for optimizer
        :@param: policy delay how often update  critic for every actor update
        :@param: batch_size: batch_size for main network training
        :@param: tau: parametere for soft update
        :@param: gamma: discount for optimal future value
        """
        self.actions_num = actions_num
        self.states_size = states_size
        torch.manual_seed(1)
        self.critic1 = Critic(states_size + actions_num, 1).to(device)
        torch.manual_seed(11)
        self.critic2 = Critic(states_size + actions_num, 1).to(device)
        self.actor = Actor(states_size, actions_num).to(device)
        self.target_critic1 = Critic(states_size + actions_num, 1).to(device)
        self.target_critic2 = Critic(states_size + actions_num, 1).to(device)
        self.target_actor = Actor(states_size, actions_num).to(device)
        self.noise = noise
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), learning_rate_critic1)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), learning_rate_critic2)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), learning_rate_actor)
        self.memory_buffer: deque = deque(maxlen=buffer_size)
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.current_steps = 0
        self.steps = 0
        self.update_num = 0
        self.rewards: List[float] = []
        self.total_rewards: List[float] = []
        self.average_rewards: deque = deque(maxlen=20)

    def step(self, state: list, action: int, reward: float, next_state: list, done: bool) -> None:

        if not done:
            self.last_reward = reward
        self.noise.evolve_state()
        self.rewards.append(reward)
        self.__add_to_memory_buffer(state, action, reward, next_state, done)
        self.current_steps += 1
        if len(self.memory_buffer) > self.batch_size:
            sample = self.__sample_memory()
            self.learn(sample)

    def act(self, state: list) -> np.array:
        """
        Returns agent action.
        """
        state_tensor = torch.tensor(state)
        return self.noise.get_action(self.actor(state_tensor).detach().numpy())

    def learn(self, sample: List[tuple]) -> None:
        """
        Learn from replay buffer and update target network.

        @param sample: list of tuples randomly choosen from memory (replay buffer)
        """
        # Convert samples to tensors
        states, actions_list, rewards, next_states, dones = map(list, zip(*sample))
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions_list).unsqueeze(-1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        temp = [1 if done is True else 0 for done in dones]
        dones_bin = torch.FloatTensor(temp).unsqueeze(1).to(device)
        criterion = torch.nn.MSELoss()
        next_actions = self.noise.get_action(self.target_actor(next_states).detach()).float()
        next_state_and_action = torch.cat((next_states, next_actions), 1)
        next_reward_1 = self.target_critic1(next_state_and_action)
        next_reward_2 = self.target_critic2(next_state_and_action)
        y_pred = rewards + self.gamma * (1 - dones_bin) * torch.minimum(next_reward_1, next_reward_2)
        state_and_action = torch.cat((states, actions.squeeze(-1)), 1)
        critic_loss_1 = criterion(self.critic1(state_and_action), y_pred.detach())
        critic_loss_2 = criterion(self.critic2(state_and_action), y_pred.detach())

        # update critics
        self.critic1_optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic_loss_2.backward()
        self.critic2_optimizer.step()

        if self.update_num % self.policy_delay == 0:
            state_and_action = torch.cat((states, self.actor(states).squeeze(-1)), 1)
            policy_loss = -self.critic1(state_and_action).mean()

            # update actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            # soft update of targets
            self.update_target()
        self.update_num += 1

    def update_target(self) -> None:
        """
        Target networks soft update.
        """
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

        for target_param, local_param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

        for target_param, local_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def show_progress(self, episode_number: int) -> None:

        if len(self.rewards) != 0:
            total_reward = sum(self.rewards)
        else:
            total_reward = 0.0
        self.total_rewards.append(total_reward)
        avg_reward = sum(self.total_rewards) / len(self.total_rewards)
        print(
            "#################################### "
            "\n# Episode no.:  "
            + str(episode_number)
            + f"\n# Reward:  {total_reward:.4f}"
            + f"\n# Avg reward: {avg_reward:.4f}"
            + "\n####################################\n"
        )
        self.rewards = []
        self.average_rewards.append(avg_reward)

    def __add_to_memory_buffer(self, state: list, action: int, reward: float, next_state: list, done: bool) -> None:
        """
         Add state, actions, reward, next states and done to memory buffer.
        @param state: Current state.
        @param action: Taken action.
        @param reward: Reward for taken action.
        @param next_state: Next state.
        @param done: Episode ended.
        @return: None
        """
        self.memory_buffer.append((state, action, reward, next_state, done))

    def __sample_memory(self) -> list:
        """
        Take batch_size random samples from memory
        @return: list of samples
        """
        return random.sample(self.memory_buffer, k=self.batch_size)

    def save(
        self,
        episodes: int,
        path: str = "checkpoints",
        filename_actor: str = None,
        filename_critic1: str = None,
        filename_critic2: str = None,
    ) -> None:
        """
        Save checkpoints.
        @param episodes: Number of episodes.
        @param path: Path to folder with checkpoints.
        @param filename_actor: Filename of actor checkpoints.
        @param filename_critic1: Filename of first critic checkpoints.
        @param filename_critic2: Filename of second critic checkpoints.
        @return: None
        """
        if not os.path.exists(path):
            os.makedirs(path)

        if filename_critic1 is None:
            filename_critic1 = "Critic_1" + "_td3_epi_" + str(episodes) + ".chpkt"
        else:
            filename_critic1 = filename_critic1 + "_td3_critic1" + "_epi_" + str(episodes) + ".chpkt"

        if filename_critic2 is None:
            filename_critic2 = "Critic_2" + "_td3_epi_" + str(episodes) + ".chpkt"
        else:
            filename_critic2 = filename_critic2 + "_td3_critic2" + "_epi_" + str(episodes) + ".chpkt"

        if filename_actor is None:
            filename_actor = "Actor" + "_td3_epi_" + str(episodes) + ".chpkt"
        else:
            filename_actor = filename_actor + "_td3_actor" + "_epi" + str(episodes) + ".chpkt"

        torch.save(self.actor.state_dict(), os.path.join(path, filename_actor))
        torch.save(self.critic1.state_dict(), os.path.join(path, filename_critic1))
        torch.save(self.critic2.state_dict(), os.path.join(path, filename_critic2))

    def load(self, filename_actor: str, filename_critic_1: str, filename_critic_2: str, path: str = "./") -> None:
        self.actor.load_state_dict(torch.load(os.path.join(path, filename_actor)))
        self.critic1.load_state_dict(torch.load(os.path.join(path, filename_critic_1)))
        self.critic2.load_state_dict(torch.load(os.path.join(path, filename_critic_2)))


def train_td3(n_episodes: int = 1000) -> None:
    """
    Initialize and train agent, discretize action space.
    @param n_episodes: total number of episode to perform
    @return : None
    """
    env = gym.make("LineFollower-v0")
    noise = OUNoise(action_space=env.action_space)
    agent = TD3Agent(states_size=16, actions_num=2, noise=noise)
    for i in range(n_episodes):
        state = env.reset()
        noise.reset()
        reward_sum = 0
        done = False
        while not done:
            action = agent.act(state)
            noise.evolve_state()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
        agent.show_progress(i)
        if i % 10 == 0 and i > 0:
            agent.save(
                i,
                filename_actor="my_model_actor",
                filename_critic1="my_model_critic1",
                filename_critic2="my_model_critic2",
            )


def test_td3(filename_actor: str, filename_critic_1: str, filename_critic_2: str, path: str = "checkpoints") -> None:
    """
    Test trained model
    """
    env = LineFollowerEnv()
    torch.manual_seed(1)
    ounoise = OUNoise(env.action_space)
    agent = TD3Agent(states_size=16, actions_num=2, noise=ounoise)
    agent.load(filename_actor, filename_critic_1, filename_critic_2, path)
    rewards = []
    for i in range(20):
        state = env.reset()
        agent.noise.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
        print("Reward: " + str(total_reward))
        rewards.append(total_reward)
    print("Average reward: " + str(sum(rewards) / len(rewards)))


if __name__ == "__main__":
    train_td3()
