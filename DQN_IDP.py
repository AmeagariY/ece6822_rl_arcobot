import gym
from  DQN_IDP_agent import *
import tensorflow as tf
import matplotlib.pyplot as plt

env_name = 'InvertedDoublePendulum-v2'
model_name = "IDP_dqn.h5"

graph_name = "IDP_dqn.png"

actionsapce_size = 21
action_list = np.linspace(-1,1,actionsapce_size)

def train(graph_name, iteration_limit=10000000, max_episodes=1000,render=False):
    env = gym.make(env_name)
    env = env.unwrapped
    state_size = env.observation_space.shape[0]
    


    agent = DQNAgent(state_size, actionsapce_size, batch_size=64, load_model=False , model_name=model_name)
    steps_counts = []
    cum_reward = []
    for episode in range(max_episodes):
        state = env.reset()
        steps=0
        cum_reward.append(0)
        while True:
            if render:
                env.render()
            action = agent.action_choose(state)
            next_state, reward, done, info = env.step([action_list[action]])
            agent.memory_append(state, action, reward, next_state, done)
            cum_reward[-1] += reward
            agent.training()
            state = next_state
            steps+=1
            if done:
                break


        print("(episode: {}; steps: {}; cumulative reward: {} ; memory length: {})"
                    .format(episode, steps, cum_reward[-1], len(agent.memory)))
        steps_counts.append(steps)
        
        if episode % 100 == 0:
            agent.model.save(model_name)
            np.save("IDP_dqn_steps_counts.npy", steps_counts)
            np.save("IDP_dqn_cum_reward.npy", cum_reward)
            draw_plot(cum_reward, filename=graph_name)

def draw_plot(steps_counts, filename):

    plt.plot(steps_counts)
    plt.title('Inverted Double Pendulum DQN Iteration vs Episodes')
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Episodes')
    plt.savefig(filename)
    #plt.show()

    

def test(graph_name, max_iter=2000, max_episodes=500,render=False):
    pass

if __name__ == '__main__':
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train(graph_name,max_episodes=50000)
    #test(test_graph_name)