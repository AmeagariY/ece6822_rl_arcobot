import gym
from  DQN_agent import *
import tensorflow as tf

env_name = 'Acrobot-v1'
model_name = "acrobot_dqn.h5"
train_graph_name = "acrobot_train_dqn.png"
test_graph_name = "acrobot_test_dqn.png"

def train(graph_name, iteration_limit=2000, max_episodes=500,render=False):
    env = gym.make(env_name)
    env = env.unwrapped
    state_size = env.observation_space.shape[0]
    actionsapce_size = env.action_space.n


    agent = DQNAgent(state_size, actionsapce_size, batch_size=64, load_model=False , model_name=model_name)

    for episode in range(max_episodes):
        state = env.reset()
        for steps in range(iteration_limit):
            if render:
                env.render()
            action = agent.action_choose(state)
            next_state, reward, done, info = env.step(action)
            agent.memory_append(state, action, reward, next_state, done)
            agent.training()
            state = next_state
            if done:
                break


        print("(episode: {}; steps: {}; memory length: {})"
                    .format(episode, steps, len(agent.memory)))
        
        if episode % 10 == 0:
            agent.model.save(model_name)

def test(graph_name, max_iter=2000, max_episodes=500,render=False):
    pass

if __name__ == '__main__':
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train(train_graph_name,max_episodes=500)
    #test(test_graph_name)