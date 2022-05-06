from fileinput import filename
import numpy as np
import gym
from QLearning_agent import *
import matplotlib.pyplot as plt

env_name = "Acrobot-v1"
model_path = "./acrobot_Q_learning4L.npy"
graph_name="acrobot_Q_learning4L.png"
range_high=[1,1,1,1,12.567,28.274]
range_low=[-1,-1,-1,-1,-12.567,-28.274]
discrete_bins=[5,5,5,5,10,10]
discrete_bins=[i+1 for i in range(len(discrete_bins))]


def train(max_iter=2000, max_episodes=500,render=False):
    env=gym.make(env_name)
    env=env.unwrapped
    state_size=env.observation_space.shape[0]
    action_size=env.action_space.n
    agent=QLearningAgent(state_size,action_size,discrete_bins)
    #agent.load_model(model_path)
    steps_counts=[]
    for episode in range(max_episodes):
        state=env.reset()
        state=state_discretization(state,state_size,range_high,range_low,discrete_bins)
        steps=0
        while True:
            if render:
                env.render()
            action=agent.action_choose(state)
            next_state,reward,done,info=env.step(action)
            nstate_discreted=state_discretization(next_state,state_size,range_high,range_low,discrete_bins)
            agent.train(state,action,reward,nstate_discreted,done)
            state=nstate_discreted
            steps+=1
            if done:
                break
        print("(episode: {}; steps: {})"
                    .format(episode, steps))
        steps_counts.append(steps)
        if episode % 50 == 0:
            agent.save_model(model_path)
    draw_plot(steps_counts,graph_name)

def draw_plot(steps_counts, filename):

    plt.plot(steps_counts)
    plt.title('Acrobot Q-Learning Iteration vs Episodes')
    plt.ylabel('Iterations')
    plt.xlabel('Episodes')
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    train(max_episodes=500,render=False)