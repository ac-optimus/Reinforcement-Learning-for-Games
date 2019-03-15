import gym
import time
env = gym.make('CartPole-v0')
env.reset()
for i in range(10000):
    env.render()
    time.sleep(0.01)
    print(i)
    env.step(env.action_space.sample())