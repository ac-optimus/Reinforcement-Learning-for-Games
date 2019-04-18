'''
	Code inspired from Advanced DQNs : Playing Pacman with Deep Reinforcement Learning
	https://towardsdatascience.com/advanced-dqns-playing-pac-man-with-deep-reinforcement-learning-3ffbd99e0814
'''

import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import Model
from keras.layers import Flatten, Convolution2D, Input, Dense
from keras.optimizers import Adam
import keras.backend as K
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint,FileLogger
import os
from PIL import Image

env_name = 'MsPacman-v0'
env = gym.make(env_name)
env.reset()

'''Observation Space = Box(210,160,3) - Pixel Values from frame
Action Space = Discrete(9) - Each action performed for a duration of k frames. 
k is uniformly sampled from {2,3,4}'''

def random_games(n):
    for episode in range(n):
        env.reset()
        while True:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
#             observation = cv2.cvtColor(observation,cv2.COLOR_BGR2GRAY)
#             observation = cv2.resize(observation,(84,84))
            if done:
                break

# random_games(3)

class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(FRAME_SIZE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == FRAME_SIZE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

FRAME_SIZE = (84,84)
WINDOW_SIZE = 4
n_actions = env.action_space.n

np.random.seed(0)
env.seed(0)

frame = Input(shape=(WINDOW_SIZE,FRAME_SIZE[0],FRAME_SIZE[1]))
cv1 = Convolution2D(32, kernel_size=(8,8), strides=4, activation='relu', data_format='channels_first')(frame)
cv2 = Convolution2D(64, kernel_size=(4,4), strides=2, activation='relu', data_format='channels_first')(cv1)
cv3 = Convolution2D(64, kernel_size=(3,3), strides=1, activation='relu', data_format='channels_first')(cv2)
dense= Flatten()(cv3)
dense = Dense(512, activation='relu')(dense)
buttons = Dense(n_actions, activation='linear')(dense)
model = Model(inputs=frame,outputs=buttons)
print(model.summary())

memory = SequentialMemory(limit=1000000, window_length=WINDOW_SIZE)

processor = AtariProcessor()

policy = policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1250000)


dqn = DQNAgent(model=model, nb_actions=n_actions, policy=policy, memory=memory,
               processor=processor, enable_double_dqn=False, enable_dueling_network=False, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)

dqn.compile(Adam(lr=.00025), metrics=['mae'])

# folder_path = './model_saves/Vanilla/'

weights_filename = 'dqn_{}_weights.h5f'.format(env_name)
checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
log_filename = 'dqn_{}_log.json'.format(env_name)
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
callbacks += [FileLogger(log_filename, interval=100)]
dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

# After training is done, we save the final weights one more time.
dqn.save_weights(weights_filename, overwrite=True)
