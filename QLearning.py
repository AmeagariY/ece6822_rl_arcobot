import numpy as np
import gym
from QLearning_agent import *

env_name = "Acrobot-v1"
model_path = "./acrobot_Q_learning_large.npy"
range_high=[1,1,1,1,12.567,28.274]
range_low=[-1,-1,-1,-1,-12.567,-28.274]
dicrete_bins=[20,20,20,20,30,30]

def train(max_iter=2000, max_episodes=500,render=False):
    env=gym.make(env_name)
    env=env.unwrapped
    state_size=env.observation_space.shape[0]
    action_size=env.action_space.n
    agent=QLearningAgent(state_size,action_size,dicrete_bins)
    #agent.load_model(model_path)
    for episode in range(max_episodes):
        state=env.reset()
        state=state_discretization(state,state_size,range_high,range_low,dicrete_bins)
        for steps in range(max_iter):
            if render:
                env.render()
            action=agent.action_choose(state)
            next_state,reward,done,info=env.step(action)
            nstate_discreted=state_discretization(next_state,state_size,range_high,range_low,dicrete_bins)
            agent.train(state,action,reward,nstate_discreted,done)
            state=nstate_discreted
            if done:
                break
        print("(episode: {}; steps: {})"
                    .format(episode, steps))
        if episode % 50 == 0:
            agent.save_model(model_path)

if __name__ == '__main__':
    train(max_episodes=5000,render=False)