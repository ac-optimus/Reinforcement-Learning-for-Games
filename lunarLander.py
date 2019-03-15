#inspired from gym official website
#random agent
import time
import gym
env = gym.make('LunarLander-v2')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
       # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        time.sleep(0.01)
        if done:     #done is returned as true when termination condition(lander touches ground) occurs
            print("Episode finished after {} timesteps".format(t+1))
            break