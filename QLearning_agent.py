#%%
import numpy as np

class QLearningAgent:
    def __init__(self, statespace_size, actionspace_size, discrete_bins, epsilon=0.99, epsilon_decacy=0.999, epsilon_min=0.1, alpha=0.1, gamma=0.9):
        self.Q = np.zeros(discrete_bins+[actionspace_size])
        self.actionspace_size = actionspace_size
        self.epsilon = epsilon
        self.epsilon_decacy = epsilon_decacy
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma

    def action_choose(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.actionspace_size)
        else:
            return np.argmax(self.Q[tuple(state)])
    
    def train(self, state, action, reward, next_state,done):
        if done:
            self.Q[tuple(state)][action]=reward
        else:
            q_predict = self.Q[tuple(state)][action]
            q_target = reward + self.gamma * np.max(self.Q[tuple(next_state)])
            self.Q[tuple(state)][action] += self.alpha * (q_target - q_predict)
        self.epsilon = self.epsilon * self.epsilon_decacy if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_model(self, path):
        np.save(path, self.Q)

    def load_model(self, path):
        self.Q = np.load(path)

def state_discretization(state_pre,statespace_size, range_high, range_low, discrete_bins):
    state_discrete = np.zeros(statespace_size,dtype=int)
    for i in range(statespace_size):
        state_discrete[i] = int(np.digitize(state_pre[i], np.linspace(range_low[i], range_high[i], discrete_bins[i]*2-1))//2)
    return state_discrete


