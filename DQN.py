import gym
from  DQN_agent import *
import tensorflow as tf
import matplotlib.pyplot as plt

env_name = 'Acrobot-v1'
model_name = "acrobot_dqn.h5"

graph_name = "acrobot_dqn.png"

def train(graph_name, iteration_limit=2000, max_episodes=500,render=False):
    env = gym.make(env_name)
    env = env.unwrapped
    state_size = env.observation_space.shape[0]
    actionsapce_size = env.action_space.n


    agent = DQNAgent(state_size, actionsapce_size, batch_size=64, load_model=False , model_name=model_name)
    steps_counts = []
    for episode in range(max_episodes):
        state = env.reset()
        steps=0
        while True:
            if render:
                env.render()
            action = agent.action_choose(state)
            next_state, reward, done, info = env.step(action)
            agent.memory_append(state, action, reward, next_state, done)
            agent.training()
            state = next_state
            steps+=1
            if done:
                break


        print("(episode: {}; steps: {}; memory length: {})"
                    .format(episode, steps, len(agent.memory)))
        steps_counts.append(steps)
        
        if episode % 10 == 0:
            agent.model.save(model_name)
    draw_plot(steps_counts, filename=graph_name)

def draw_plot(steps_counts, filename):

    plt.plot(steps_counts)
    plt.title('Acrobot DQN Iteration vs Episodes')
    plt.ylabel('Iterations')
    plt.xlabel('Episodes')
    plt.savefig(filename)
    plt.show()

    

def test(graph_name, max_iter=2000, max_episodes=500,render=False):
    pass

if __name__ == '__main__':
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train(graph_name,max_episodes=100)
    #test(test_graph_name)