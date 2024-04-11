import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


class Qlearning:
    def __init__(self, state_size, action_size, alpha, gamma, epsilon):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))

    def update(self, state, action, reward, next_state):
        delta = reward + self.gamma * \
            np.max(self.q_table[next_state]) - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * delta

    def reset(self):
        self.q_table = np.zeros((self.state_size, self.action_size))

    def plot_heatmap(self, savefig_folder):
        q_table_max = np.max(self.q_table, axis=1).reshape(4, 4)
        sns.heatmap(q_table_max, annot=True, cmap='coolwarm', fmt=".2f")
        plt.savefig(savefig_folder / "qlearning_heatmap.png")
        plt.show()


class Epsgreedy:
    def __init__(self, epsilon, action_space):
        self.epsilon = epsilon
        self.action_space = action_space

    def sample_action(self, q_values, greedy=False):
        # exploration
        if random.random() < self.epsilon and not greedy:
            return np.random.randint(0, self.action_space, dtype=int)
        # exploitation
        indices = np.where(q_values == np.max(q_values))[0]
        if len(indices) == 1:
            return indices[0]
        random_index = np.random.randint(0, len(indices))
        return indices[random_index]

    def decay_epsilon(self):
        self.epsilon *= 0.999


if __name__ == "__main__":
    agent = Epsgreedy(epsilon=0.1, action_space=4)
    print(agent.sample_action(np.array([1, 2, 3, 4]), greedy=True))
    print(agent.sample_action(np.array([1, 1, 1, 1]), greedy=True))
    print(agent.sample_action(np.array([1, 2, 3, 5]), greedy=True))
    print(agent.sample_action(np.array([1, 1, 2, 2]), greedy=True))
