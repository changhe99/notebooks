import gymnasium as gym
import numpy as np
from typing import NamedTuple, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

from utilities import ReplayBuffer
from agent import Qlearning, Epsgreedy


class Params(NamedTuple):
    name: str
    total_episodes: int
    learning_rate: float
    gamma: float
    epsilon: float
    map_size: int
    savefig_folder: Path


class Trainer:
    def __init__(self, env, test_env, agent, critic, num_episodes):
        self.env = env
        self.test_env = test_env
        self.agent = agent
        self.critic = critic
        self.replay_buffer = ReplayBuffer(capacity=1000)
        self.num_episodes = num_episodes

    def train(self):
        for i in range(self.num_episodes):
            obs, info = self.env.reset()
            terminated = False
            while not terminated:
                qvalues = self.critic.q_table[obs]
                action = self.agent.sample_action(qvalues)
                next_obs, reward, terminated, truncated, info = self.env.step(
                    action)
                self.replay_buffer.push(obs, action, next_obs, reward)
                if terminated or truncated:
                    obs, info = self.env.reset()
                    break
                obs = next_obs
            if len(self.replay_buffer) > 64:
                self.update_critic()
            self.agent.decay_epsilon()

    def update_critic(self):
        batch = self.replay_buffer.sample(batch_size=64)
        for transition in batch:
            state, action, next_state, reward = transition
            self.critic.update(state, action, reward, next_state)

    def test(self, num_episodes=1):
        for i in range(num_episodes):
            obs, info = self.test_env.reset()
            terminated = False
            while not terminated:
                qvalues = self.critic.q_table[obs]
                action = self.agent.sample_action(qvalues, greedy=True)
                next_obs, reward, terminated, truncated, info = self.test_env.step(
                    action)
                if terminated or truncated:
                    obs, info = self.test_env.reset()
                    break
                obs = next_obs


if __name__ == '__main__':
    params = Params(name='FrozenLake-v1', total_episodes=2000, learning_rate=0.1, gamma=0.99,
                    epsilon=0.5, map_size=4, savefig_folder=Path('figures'))
    params.savefig_folder.mkdir(exist_ok=True)

    # create environment
    env = gym.make(params.name, render_mode='rgb_array')
    test_env = gym.make(params.name, render_mode='human')

    learner = Qlearning(state_size=env.observation_space.n,
                        action_size=env.action_space.n,
                        alpha=params.learning_rate,
                        gamma=params.gamma,
                        epsilon=params.epsilon)
    actor = Epsgreedy(epsilon=params.epsilon, action_space=env.action_space.n)

    trainer = Trainer(env, test_env, actor, learner,
                      num_episodes=params.total_episodes)
    trainer.train()
    learner.plot_heatmap(params.savefig_folder)
    trainer.test(num_episodes=1)
