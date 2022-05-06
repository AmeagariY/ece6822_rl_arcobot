import gym
from  DQN_IDP_agent import *
import tensorflow as tf
import matplotlib.pyplot as plt

def draw_plot(cum_reward,cum_reward_100ave, filename):

    plt.plot(cum_reward,label="Cumulative Reward")
    plt.plot(cum_reward_100ave,label="1000-episode average")
    plt.title('Inverted Double Pendulum DQN Iteration vs Episodes')
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Episodes')
    
    plt.legend()
    plt.savefig(filename)
    #plt.show()

graph_name = "IDP_dqn_ave1000.png"
cum_reward=np.load("IDP_dqn_cum_reward.npy")
cumsum_vec = np.cumsum(np.insert(cum_reward, 0, 0)) 
window_width = 1000
cum_reward_100ave = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
draw_plot(cum_reward,cum_reward_100ave, filename=graph_name)
print(cum_reward_100ave[-1])
