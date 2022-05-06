from keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.callbacks import Callback
import random
import numpy as np
import os
from collections import deque

class DQNAgent:
    def __init__(self, state_size, actionsapce_size,
                load_model=False, model_name="dqn_model.h5",
                gamma=0.99, learning_rate=0.001,
                epsilon=1.0, epsilon_decay=0.999,
                epsilon_min=0.05, batch_size=64, memory_size=2000,
                ):
        
        self.state_size = state_size
        self.actionsapce_size = actionsapce_size



        self.load_model = load_model
        self.model_name = model_name

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay  
        self.epsilon_min = epsilon_min  
        self.batch_size = batch_size


        self.memory = deque(maxlen=memory_size)


        self.model = self.model_build()

    def model_build(self, units=512):
        if self.load_model and os.path.exists(self.model_name):
            model = load_model(self.model_name)
        else:
            model = Sequential()
            #model(Input(shape=(self.state_size,)))
            model.add(Dense(units, input_dim=self.state_size, activation='sigmoid'))
            model.add(Dense(units, activation='relu'))
            model.add(Dense(units, activation='relu'))
            model.add(Dense(self.actionsapce_size, activation='linear'))
            model.summary()
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def action_choose(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actionsapce_size)
        else:
            return np.argmax(self.model.predict(np.reshape(state, [1, self.state_size]))[0])
    
    def memory_append(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def training(self):
        batch_size = min(self.batch_size, len(self.memory))
        batch = random.sample(self.memory, batch_size)
        states= np.array([i[0] for i in batch])
        Q_next=self.model.predict(np.array([i[3] for i in batch]), batch_size=batch_size)
        Q=self.model.predict(states, batch_size=batch_size)
        for i in range(len(batch)):
            if batch[i][4]:
                Q[i][batch[i][1]]=batch[i][2]
            else:
                Q[i][batch[i][1]]=batch[i][2]+self.gamma*np.max(Q_next[i])
        self.model.fit(states, Q, batch_size=batch_size, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

