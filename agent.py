import numpy as np
import random


class QLearningAgent:

    def __init__(self, grid_size, learning_rate=0.1, discount_factor=0.9, epsilon=1.0):

        # learning rate (how fast agent learns)
        self.lr = learning_rate

        # discount factor (importance of future reward)
        self.gamma = discount_factor

        # exploration rate
        self.epsilon = epsilon

        # number of actions
        self.actions = 4

        # Q table initialization
        self.q_table = np.zeros((grid_size, grid_size, self.actions))


    def choose_action(self, state):

        # epsilon-greedy policy

        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.actions - 1)

        row, col = state
        return np.argmax(self.q_table[row, col])


    def update_q(self, state, action, reward, next_state):

        row, col = state
        next_row, next_col = next_state

        best_next_action = np.max(self.q_table[next_row, next_col])

        current_q = self.q_table[row, col, action]

        new_q = current_q + self.lr * (
            reward + self.gamma * best_next_action - current_q
        )

        self.q_table[row, col, action] = new_q


    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):

        if self.epsilon > min_epsilon:
            self.epsilon *= decay_rate