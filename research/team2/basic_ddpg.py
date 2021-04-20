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
import os.path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OUNoise:
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    Based on the rllab implementation.
    class from: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    """

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
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
        Resets the noise generator.
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

    def get_action(self, action: np.array, t: int = 0) -> np.array:
        """
        Returns an action with added noise.
        @param action: Tensor with actions.
        @param t: Time step.
        @return: Action + noise
        """
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class Actor(nn.Module):
    def __init__(self, states_size, actions_num):
        """
        Create
        @param states_size: state vector size
        @param actions_num: size of action space
        """
        super().__init__()
        hidden_size = 256
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
        Create
        @param states_size: state vector size
        @param actions_num: size of action space
        """
        super().__init__()
        hidden_size = 256
        self.fc1 = nn.Linear(states_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, actions_num)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


class DDPGAgent:
    """
    Implements DQN algorithm and agent.
    """

    def __init__(
        self,
        states_size: int,
        actions_num: int,
        buffer_size: int = 100000,
        learning_rate_actor: float = 0.001,
        learning_rate_critic: float = 0.001,
        batch_size: int = 32,
        tau: float = 0.01,
        gamma: float = 0.99,
    ) -> None:
        """
        Target network soft update.
        :@param: states_size: size of state vector ( 16 )
        :@param: actions_num: size of action space
        :@param: buffer_size: size of replay buffer, agents memory
        :@param: learning_rate: learning rate for optimizer
        :@param: update_cooldown: how often target network update takes place
        :@param: batch_size: batch_size for main network training
        :@param: tau: parametere for soft update
        :@param: gamma: discount for optimal future value
        :@param: epslion: probability of greedy behaviour
        """
        self.actions_num = actions_num
        self.states_size = states_size
        self.critic = Critic(states_size + actions_num, 1).to(device)
        self.critic_criterion = torch.nn.MSELoss()
        self.actor = Actor(states_size, actions_num).to(device)
        self.target_critic = Critic(states_size + actions_num, 1).to(device)
        self.target_actor = Actor(states_size, actions_num).to(device)
        self.lr_actor = learning_rate_actor
        self.lr_critic = learning_rate_critic
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.memory_buffer: deque = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.current_steps = 0
        self.steps = 0
        self.total_rewards: List[float] = []
        self.rewards: List[float] = []
        self.last_reward = 0.0
        self.avg_rewards: deque = deque(maxlen=20)

    def step(self, state: list, action: int, reward: float, next_state: list, done: bool) -> None:
        """
        Agent step, memory update and learn.
        """
        self.rewards.append(reward)
        self.__add_to_memory_buffer(state, action, reward, next_state, done)
        self.current_steps += 1
        if len(self.memory_buffer) > self.batch_size:
            sample = self.__sample_memory()
            self.learn(sample)

    def act(self, state: list) -> torch.Tensor:
        """
        Take an action

        :@param state: list of 8 points returned from enviroment
        :@return: tensor of actions
        """
        state_tensor = torch.tensor(state)
        return self.actor(state_tensor).detach()

    def learn(self, sample: List[tuple]) -> None:
        """
        Learn from replay buffer and upadte target network.
        :@param sample: List of random tuples from replay buffer.
        :@return: None
        """
        # convert sample to tensor
        states, actions_list, rewards, next_states, dones = map(list, zip(*sample))
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions_list).unsqueeze(-1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        state_and_action = torch.cat((states, actions.squeeze(-1)), 1)
        q_val = self.critic(state_and_action)
        next_actions = self.target_actor(next_states).squeeze(-1)  # OpenAI adds here noise
        next_state_and_action = torch.cat((next_states, next_actions), 1)
        next_reward = self.target_critic(next_state_and_action)
        y_pred = rewards + self.gamma * next_reward  # value predicted by target network
        # update critic
        critic_loss = self.critic_criterion(q_val, y_pred.detach())
        state_and_action = torch.cat((states, self.actor(states).squeeze(-1)), 1)
        policy_loss = -self.critic(state_and_action).mean()
        self.last_loss = policy_loss
        # perform updates
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # soft update
        self.update_target()

    def update_target(self) -> None:
        """
        Target network soft update.
        """
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def show_progress(self, episode_number: int) -> None:
        """
        Show progress.
        """
        if len(self.rewards) != 0:
            total_reward = sum(self.rewards)
        else:
            total_reward = 0.0
        self.total_rewards.append(total_reward)
        avg_reward = sum(self.total_rewards) / len(self.total_rewards)
        self.avg_rewards.append(avg_reward)
        print(
            "#################################### "
            "\n# Episode no.:  "
            + str(episode_number)
            + f"\n# Reward:  {total_reward:.4f}"
            + f"\n# Avg reward: {avg_reward:.4f}"
            + "\n####################################\n"
        )
        self.rewards = []

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
        self, episodes: int, path: str = "checkpoints", filename_actor: str = None, filename_critic: str = None
    ) -> None:
        """
        Save model checkpoints
        @param episodes: number of episodes
        @param path: folder where to save checkpoints
        @param filename_actor: actor checkpoints filename
        @param filename_critic: critic checkpoints filename
        @return: None
        """
        if not os.path.exists(path):
            os.makedirs(path)

        if filename_actor is None:
            filename_actor = "Actor_ddpg" + "_epi-" + str(episodes) + ".chpkt"
        else:
            filename_actor = filename_actor + "_ddpg_epi_" + str(episodes) + ".chpkt"

        if filename_critic is None:
            filename_critic = "Critic_ddpg_" + "_ddpg_epi-" + str(episodes) + ".chpkt"
        else:
            filename_critic = filename_critic + "_ddpg_epi_" + str(episodes) + ".chpkt"

        torch.save(self.critic.state_dict(), os.path.join(path, filename_critic))
        torch.save(self.actor.state_dict(), os.path.join(path, filename_actor))

    def load(self, filename_critic: str, filename_actor: str, path: str) -> None:
        """
        Load model checkpoints.
        @param filename_critic: filename of critic checkpoints
        @param filename_actor: filename of actor checkpoints
        @param path: folder for checkpoints
        @return: None
        """
        self.actor.load_state_dict(torch.load(os.path.join(path, filename_critic)))
        self.critic.load_state_dict(torch.load(os.path.join(path, filename_actor)))


def train_ddpg(n_episodes: int = 1000) -> None:
    """
    Initialize and train agent, discretize action space.
    @param n_episodes: total number of episode to perform
    """
    env = LineFollowerEnv(gui=False)
    torch.manual_seed(1)
    action_space = 2
    state_space = 16
    agent = DDPGAgent(state_space, action_space)
    noise = OUNoise(action_space=env.action_space)
    for i in range(n_episodes):
        state = env.reset()
        noise.reset()
        reward_sum = 0
        for j in range(1000):
            action = noise.get_action(agent.act(state).numpy(), j)
            noise.evolve_state()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.show_progress(i)
        if i % 100 == 0 and i > 0:
            agent.save(i, filename_actor="my_model_actor", filename_critic="my_model_critic")


def test_ddpg(filename_critic: str, filename_actor: str, path: str = "./checkpoints") -> None:
    """
    Tests model.
    @param filename_critic: Critic checkpoints filename.
    @param filename_actor: Actor checkpoints filename.
    @param path: Path to folder with checkpoints.
    @return: None
    """
    env = LineFollowerEnv()
    torch.manual_seed(1)
    agent = DDPGAgent(states_size=16, actions_num=2)
    agent.load(filename_critic=filename_critic, filename_actor=filename_actor, path=path)
    rewards = []
    for i in range(20):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state).numpy()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
        print("Reward: " + str(total_reward))
        rewards.append(total_reward)
    print("Average reward: " + str(sum(rewards) / len(rewards)))


if __name__ == "__main__":
    train_ddpg()
