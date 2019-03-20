"""Inspired by https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578"""

import gym
import numpy as np
import math
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time
from plotUtils import plot_

class ObsDesc:
    def __init__(self, env, buckets=(1, 1, 6, 12,), up=None, low=None):
        self.env = env
        self.buckets = buckets
        if up is None or low is None:
            self.up  = [self.env.observation_space.high[0], 0.5,
                        self.env.observation_space.high[2], math.radians(50)]
            self.low = [self.env.observation_space.low[0], -0.5,
                        self.env.observation_space.low[2], -math.radians(50)]
        else:
            self.up = up
            self.low = low
        assert len(self.up) == len(self.low)

    def discretize(self, observation):
        ratios = [(observation[i] + abs(self.low[i])) / (self.up[i] - self.low[i]) for i in range(len(self.low))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(observation))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(observation))]
        return tuple(new_obs)


class QCartPoleSolver():
    def __init__(self, n_episodes=1000, n_win_ticks=195,
                 min_alpha=0.1, min_epsilon=0.1,
                 gamma=1.0, ada_divisor=25, max_env_steps=None,
                 quiet=False, monitor=False):
        self.n_episodes = n_episodes # training episodes 
        self.n_win_ticks = n_win_ticks # average ticks over 100 episodes required for win
        self.min_alpha = min_alpha # learning rate
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount factor
        self.ada_divisor = ada_divisor # only for development purposes
        self.quiet = quiet
        self.env = gym.make('CartPole-v0')
        self.obscretizer = ObsDesc(self.env)

        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        if monitor: self.env = gym.wrappers.Monitor(self.env, 'tmp/cartpole-1', force=True) # record results for upload

        self.Q = np.zeros(self.obscretizer.buckets + (self.env.action_space.n,))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))



    def run_naya(self):
        lis = []
        count =0
        for i in range(200):
            retrn = 0
            state = self.obscretizer.discretize(self.env.reset())
            alpha = self.get_alpha(i)
            epsilon = self.get_epsilon(i)
            done =False
            while True:
                retrn += 1
                #time.sleep(0.01)
               # self.env.render()
                #time.sleep(0.1)
                action = self.choose_action(state, epsilon)
                state_next, reward, done, info = self.env.step(action)
                state_next = self.obscretizer.discretize(state_next)
                self.Q[state][action] += alpha * \
                    (reward + self.gamma * np.max(self.Q[state_next]) - self.Q[state][action])
                retrn += 1
                state = state_next
                if done:
                    
                    lis.append(retrn)
                    if retrn >200:
                        print (count)
                        count +=1
                    print ("Epsoid: " + str(i) + "score: " + str(retrn))
                    break 
            
        plot_(np.arange(len(lis)),lis,"No of episode","Reward","Win Rate",200,count)

        # plt.plot(np.arange(len(lis_)),lis_)
        # plt.xlabel("No of episodes")
        # plt.ylabel("Return(Cummulative rewards)")
        # plt.title("Q Learning")
        # plt.show()
    def inference (self):
        lis = []
        count =0
        for i in range(200):
            retrn = 0
            state = self.obscretizer.discretize(self.env.reset())
            done =False
            while True:
                retrn += 1
                #time.sleep(0.01)
               # self.env.render()
                #time.sleep(0.1)
                action = np.argmax(self.Q[state])
                state_next, reward, done, info = self.env.step(action)
                state_next = self.obscretizer.discretize(state_next)
                # self.Q[state][action] += alpha * \
                #     (reward + self.gamma * np.max(self.Q[state_next]) - self.Q[state][action])
                retrn += 1
                state = state_next
                if done:
                    lis.append(retrn)
                    if retrn >200:
                        count +=1
                    print ("Epsoid: " + str(i) + "score: " + str(retrn))
                    break 
            
        plot_(np.arange(len(lis)),lis,"No of episode","Reward","Win Rate",200,count)


if __name__ == "__main__":
    solver = QCartPoleSolver()
    solver.run_naya()
    solver.inference()
    # gym.upload('tmp/cartpole-1', api_key='')