"""Inspired by https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578"""

import gym
import numpy as np
import math
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
# from ..plotUtils import plot_

import matplotlib.pyplot as plt
import numpy as np
epis = 15000
win_rewq  = 200
def plot_(x_, y_,x_label="Not given", y_label = "Not given", title_ = "Not given", Thresh = None,count=None):
    try:
        if Thresh != None:
            plt.plot(np.arange(len(y_)),np.full(len(y_),Thresh))
        plt.plot(x_,y_)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title_+" "+str(count)+"/"+str(Thresh))
        plt.show()
    except :
        print ("usage: plot_(x_, y_,x_label, y_label , title_ , Thresh ,count")
        

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
                 min_alpha=0.1, min_epsilon=0.25,
                 gamma=1.0, ada_divisor=25, max_env_steps=None,
                 quiet=False, monitor=False):
        self.n_episodes = n_episodes # training episodes 
        self.n_win_ticks = n_win_ticks # average ticks over 100 episodes required for win
        self.min_alpha = min_alpha # learning rate
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount factor
        self.ada_divisor = ada_divisor # only for development purposes
        self.quiet = quiet
        self.env = gym.make('LunarLander-v2')
        bb = tuple([10]*8)
        uu = [0.009812164, 1.4029007, 0.053989492, 0.13097233, 0.3719464, 0.60632604, 1.0, 1.0]
        dd = [-0.116679095, -0.043012504, -0.1512446, -1.505033, -0.06693668, -4.924051, 0.0, 0.0]
        self.obscretizer = ObsDesc(self.env, buckets=bb,
                                    up=uu,
                                    low=dd)

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
        episodsss = []
        lis = []
        count =0
        for i in range(epis):
            retrn = 0
            state = self.obscretizer.discretize(self.env.reset())
            alpha = self.get_alpha(i)
            epsilon = self.get_epsilon(i)
            done =False
            while True:
                #time.sleep(0.1)
                action = self.choose_action(state, epsilon)
                state_next, reward, done, info = self.env.step(action)
                state_next = self.obscretizer.discretize(state_next)
                self.Q[state][action] += alpha * \
                    (reward + self.gamma * np.max(self.Q[state_next]) - self.Q[state][action])
                retrn += reward
                state = state_next
                if done:
                    
                    lis.append(retrn)
                    if retrn >win_rewq:
                        count +=1
                    # print ("Episode: " + str(i) + " score: " + str(retrn))
                    episodsss.append(i)
                    break 
        
        with open("discrete_ll_train_sc_withEps"+str(epis)+".pkl", 'wb') as f:
            pickle.dump(lis, f)
        plot_(np.arange(len(lis)),lis,"No of episode","Reward","Win Rate",win_rewq,count)

        
    def inference (self):
        episodsss = []
        lis = []
        count =0
        for i in range(epis):
            retrn = 0
            state = self.obscretizer.discretize(self.env.reset())
            done =False
            while True:
                #time.sleep(0.01)
                self.env.render()
                #time.sleep(0.1)
                action = np.argmax(self.Q[state])
                state_next, reward, done, info = self.env.step(action)
                state_next = self.obscretizer.discretize(state_next)
                # self.Q[state][action] += alpha * \
                #     (reward + self.gamma * np.max(self.Q[state_next]) - self.Q[state][action])
                retrn += reward
                state = state_next
                if done:
                    lis.append(retrn)
                    if retrn >win_rewq:
                        count +=1
                    print ("Epsoid: " + str(i) + "score: " + str(retrn))
                    episodsss.append(i)
                    break
        with open("discrete_ll_inf_sc_withEps"+str(epis)+".pkl", 'wb') as f:
            pickle.dump(lis, f)
        
            
        plot_(np.arange(len(lis)),lis,"No of episode","Reward","Win Rate",win_rewq,count)

import sys
if __name__ == "__main__":
    fnmam = 'model.pkl'
    solver = QCartPoleSolver()
    if sys.argv[1] == 'train':
        print("running model")
        solver.run_naya()
        print("stroing model")
        with open(fnmam, 'wb') as f:
            pickle.dump(solver.Q, f)
        print("stored.")
    elif sys.argv[1] == 'inf':
        with open(fnmam, 'rb') as f:
            solver.Q = pickle.load(f)
        print("Loaded model")
    print("-------------------------- inference --------------------------")
    solver.inference()
    # gym.upload('tmp/cartpole-1', api_key='')
