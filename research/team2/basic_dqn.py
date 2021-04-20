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


class QNet(nn.Module):
    def __init__(self, states_size, actions_num):
        """
        Create
        @param states_size: state vector size
        @param actions_num: size of action space
        """
        super().__init__()
        hidden_size = 64
        self.fc1 = nn.Linear(states_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, actions_num)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


class DQNAgent:
    """
    Implements DQN algorithm and agent.
    """

    def __init__(
        self,
        states_size: int,
        actions_num: int,
        buffer_size: int = 10000,
        learning_rate: float = 0.0005,
        update_cooldown: int = 4,
        batch_size: int = 64,
        tau: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
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
        self.main_net = QNet(states_size, actions_num).to(device)
        self.target_net = QNet(states_size, actions_num).to(device)
        self.lr = learning_rate
        self.optimizer = optim.Adam(self.main_net.parameters(), learning_rate)
        self.memory_buffer: deque = deque(maxlen=buffer_size)
        self.update_cooldown = update_cooldown
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.current_steps = 0
        self.steps = 0
        self.last_loss = 0.0
        self.rewards: List[float] = []
        self.losses: List[float] = []
        self.last_reward = 0.0
        self.train = True

    def step(self, state: list, action: int, reward: float, next_state: list, done: bool) -> None:
        """
        Agent step, updates memory and calls learn() performed on every iteration
        :@param: state: state vector
        :@param: actions_num: size of action space
        :@param: buffer_size: size of replay buffer, agents memory
        :@param: learning_rate: learning rate for optimizer
        :@param: update_cooldown: how often target network update takes place
        :@param: batch_size: batch_size for main network training
        :@param: tau: parametere for soft update
        :@param: gamma: discount for optimal future value
        :@param: epslion: probability of greedy behaviour
        """
        if not done:
            self.last_reward = reward
        self.rewards.append(reward)
        self.__add_to_memory_buffer(state, action, reward, next_state, done)
        self.current_steps += 1
        if len(self.memory_buffer) > self.batch_size:
            if (self.current_steps + 1) % self.update_cooldown == 0:
                sample = self.__sample_memory()
                self.learn(sample)

    def act(self, state: list) -> int:
        """
        Take an action

        :param state: list of 8 points returned from enviroment
        :return: action index from action_space
        """
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.main_net.eval()
        with torch.no_grad():
            actions = self.main_net(state)
        self.main_net.train()
        if random.random() > self.epsilon:
            return int(np.argmax(actions.cpu().data.numpy()))
        else:
            return int(random.choice(np.arange(self.actions_num)))

    def answer(self, state: list) -> int:
        """
        For testing purposes, just returns best action
        @param: state: state vector
        @return: best action
        """
        self.main_net.eval()
        state = torch.FloatTensor(state)
        with torch.no_grad():
            actions = self.main_net(state)
        return int(np.argmax(actions.cpu().data.numpy()))

    def learn(self, sample: List[tuple]) -> None:
        """
        Learn from replay buffer and upadte target network.

        :param sample: list of tuples from replay buffer
        """
        states, actions, rewards, next_states, dones = map(list, zip(*sample))
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(-1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        temp = [1 if done is True else 0 for done in dones]
        dones_bin = torch.FloatTensor(temp).unsqueeze(1).to(device)
        criterion = torch.nn.MSELoss()
        self.main_net.train()
        self.target_net.eval()
        predicted_targets = self.main_net(states).gather(dim=1, index=actions)
        with torch.no_grad():
            labels_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        labels = rewards + (self.gamma * labels_next * (1 - dones_bin))
        loss = criterion(predicted_targets, labels).to(device)
        self.last_loss = loss.data.numpy()
        self.losses.append(loss.data.numpy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        self.update_target()

    def update_target(self) -> None:
        """
        Target network soft update.
        """
        for target_param, local_param in zip(self.target_net.parameters(), self.main_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

    def show_progress(self, episode_number: int) -> None:
        """
        Target network soft update.
        """
        if len(self.losses) > 0:
            avg_loss = sum(self.losses) / len(self.losses)
        else:
            avg_loss = 0.0
        if len(self.rewards) != 0:
            total_reward = sum(self.rewards)
        else:
            total_reward = 0.0
        print("####################################")
        print("# Episode no.:  " + str(episode_number))
        print(f"# Avg loss: {avg_loss:.4f}")
        print(f"# Last loss: {self.last_loss:.4f}")
        print(f"# Reward:  {total_reward:.4f}")
        print("####################################")
        self.rewards = []
        self.losses = []

    def __add_to_memory_buffer(self, state: list, action: int, reward: float, next_state: list, done: bool) -> None:
        """
        Target network soft update.
        """
        self.memory_buffer.append((state, action, reward, next_state, done))

    def __sample_memory(self) -> list:
        """
        Take batch_size random samples from memory
        """
        return random.sample(self.memory_buffer, k=self.batch_size)

    def save(self, episodes: int, path: str = "./", filename: str = None) -> None:
        """
        Save model
        """
        if filename is None:
            filename = "gmm-" + str(self.gamma) + "_eps_" + str(self.epsilon) + "_episodes-" + str(episodes) + ".chpkt"
        else:
            filename = filename + "_epi_" + str(episodes) + ".chpkt"
        torch.save(self.main_net.state_dict(), os.path.join(path, filename))

    def load(self, filename: str, path: str = "./") -> None:
        self.main_net.load_state_dict(torch.load(os.path.join(path, filename)))


def train_dqn(n_episodes: int = 1000, eps_decay: float = 0.996, eps_end: float = 0.01) -> None:
    """
    Initialize and train agent, discretize action space.
    @param n_episodes: total number of episode to perform
    @param eps_decay: decay of epsilon for every episode
    @param eps_end: end value for epsilon
    """
    env = gym.make("LineFollower-v0")
    torch.manual_seed(1)
    rotor_step = 0.25
    states_rotor = np.arange(0, 1 + rotor_step, rotor_step)
    action_space = [
        (rotor_one_state, rotor_two_state) for rotor_one_state in states_rotor for rotor_two_state in states_rotor
    ]
    agent = DQNAgent(16, len(action_space))
    for i in range(n_episodes):
        state = env.reset()
        reward_sum = 0
        done = False
        agent.epsilon = max(agent.epsilon * eps_decay, eps_end)
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action_space[action])
            reward_sum += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
        agent.show_progress(i)
        if i % 100 == 0 and i > 0:
            agent.save(i, filename="my_model")


def test_dqn(filename: str, path: str = "./") -> None:
    """
    Test trained model
    """
    env = LineFollowerEnv()
    rotor_step = 0.25
    states_rotor = np.arange(0, 1 + rotor_step, rotor_step)
    action_space = [
        (rotor_one_state, rotor_two_state) for rotor_one_state in states_rotor for rotor_two_state in states_rotor
    ]
    torch.manual_seed(1)
    agent = DQNAgent(states_size=16, actions_num=25)
    agent.load(filename, path)
    for i in range(20):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.answer(state[:16])
            next_state, reward, done, _ = env.step(action_space[action])
            state = next_state
            total_reward += reward
        print("Reward: " + str(total_reward))


if __name__ == "__main__":
    train_dqn()
