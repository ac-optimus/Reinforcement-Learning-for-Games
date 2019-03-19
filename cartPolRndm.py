#random agent
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('CartPole-v1')
lis = []
for i_episode in range(100):
    observation = env.reset()
    retrn = 0
    for t in range(100):
        env.render()
       # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        retrn += reward
        time.sleep(0.01)
        if done:     #done is returned as true when termination condition(lander touches ground) occurs
            print("Epsiod:Reward",i_episode,retrn)
            lis.append(retrn)
            break
plt.plot(np.arange(len(lis)),lis)
plt.ylabel("Return(cummulative reward)")
plt.xlabel("Epsiod #")
plt.title("Random Agent")
plt.show()