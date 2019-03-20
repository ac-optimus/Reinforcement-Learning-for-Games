#random agent
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
from plotUtils import plot_ 

env = gym.make('CartPole-v1')
lis = []
count =0
for i_episode in range(200):
    observation = env.reset()
    retrn = 0
    while True :
#        env.render()
       # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        retrn += reward
        time.sleep(0.01)
        if done:     #done is returned as true when termination condition(lander touches ground) occurs
            print("Epsiod:Reward",i_episode,retrn)
            if count >200:
                count +=1
            lis.append(retrn)
            break
    
plot_(np.arange(len(lis)),lis,"No of episode","Reward","Win Rate",200,count)

# plt.plot(np.arange(len(lis)),lis)
# plt.ylabel("Return(cummulative reward)")
# plt.xlabel("Epsoide #")
# plt.title("Random Agent")
# plt.show()