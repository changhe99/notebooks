from utilities import ReplayBuffer, Transition
from agent import DQN

from gym.wrappers import RecordVideo
import torch.optim as optim
import torch.nn as nn
import torch
import gymnasium as gym
import random
import math

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


class Trainer:
    def __init__(self, env):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.env = env
        self.n_actions = env.action_space.n
        state, info = self.env.reset()
        self.n_observations = len(state)

        self.policy_net = DQN(self.n_observations,
                              self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations,
                              self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.replay_buffer = ReplayBuffer(10000)

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + \
            (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        transitions = self.replay_buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(
            lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).max(1).values
        expected_state_action_values = (
            next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state, info = self.env.reset(seed=0)
            state = torch.tensor(state, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.replay_buffer.push(state, action, next_state, reward)

                state = next_state
                self.optimize_model()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = TAU * policy_net_state_dict[key] + (
                        1 - TAU) * target_net_state_dict[key]
                self.target_net.load_state_dict(target_net_state_dict)

    def test(self, num_episodes=1):
        env = gym.make('CartPole-v1', render_mode="rgb_array")
        env = RecordVideo(
            env,
            episode_trigger=lambda x: True,
            video_folder='video')

        # state, info = env.reset()

        # env.start_video_recorder()

        for i in range(num_episodes):
            state, info = env.reset(seed=random.randint(0, 1000))
            state = torch.tensor(state, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
            done = False
            while not done:
                # env.render()
                action = self.target_net(state).max(1)[1].view(1, 1)
                next_state, reward, terminated, truncated, _ = env.step(
                    action.item())
                state = torch.tensor(next_state, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
                done = terminated or truncated
                if done:
                    break

        # env.close_video_recorder()
        env.close()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    trainer = Trainer(env)
    trainer.train(1000)
    env.close()
    print('Training complete')

    trainer.test(5)
