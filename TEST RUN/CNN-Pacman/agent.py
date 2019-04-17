"""
Class which encapsulates the learning process and I/O for decision-making

Prabhat Rayapati (pr2sn)
Zack Verham (zdv8rb)
Deep Learning for Computer Graphics (Fall 2016)
Final Project
"""


import random
import os
import operator

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import sgd

from skimage import io, color, transform

import numpy as np

class Agent:

    def __init__(self, epsilon=0.1, experience_replay_capacity=500, minibatch_size=20, learning_rate=0.1):

        #note: this is a modification which ovverrides our original implementation
        #to allow for a decreasing epsilon over time. Therefore, ignore the
        #epsilon value passed into the constructor (kept in constructor so that
        #the decreasing epsilon code can be removed without any significant
        #changes)
        self.epsilon = 0.1
        self.max_frames = 100000
        self.cur_frames = 0

        self.minibatch_size = minibatch_size

        self.learning_rate = learning_rate

        #defined in paper
        self.processed_image_dim = 84

        #initialize replay memory D to capacity N
        self.experience_replay_capacity = experience_replay_capacity
        self.experience_replay_counter = 0
        self.min_experiences = 0
        self.experience_replay = []
        for i in range(experience_replay_capacity):
            self.experience_replay.append({"is_valid": False})

        #approximates  Q function - following implementation detailed in paper
        model = Sequential()
        model.add(Convolution2D(16, kernel_size=(8,8), strides=(4, 4),data_format='channels_first', input_shape=(4, self.processed_image_dim, self.processed_image_dim)))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, kernel_size=(4, 4), strides=(2, 2),data_format='channels_first' ))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(8))
        model.compile(loss='mse', optimizer='rmsprop')
        print(model.summary())
        self.model = model

    def save_model(self, iteration):
        """
        Helper function for saving the model at a particular iteration
        """
        self.model.save("tmp/model/%s_1.h5" % iteration)

    def load_model(self, path):
        """
        Helper function for loading a previously saved model
        """
        self.model = load_model(path)

    def append_experience_replay_example(self, experience_replay_example):
        """
        Add an experience replay example to our agent's replay memory. If
        memory is full, overwrite previous examples, starting with the oldest
        """

        #if memory is full, reset index
        if self.experience_replay_counter >= self.experience_replay_capacity:
            self.experience_replay_counter = 0

        #track if enough experiences have occurred for a minibatch update
        if self.min_experiences < self.experience_replay_capacity:
            self.min_experiences += 1

        #convert to numpy arrays before pasing into memory
        experience_replay_example["current_state"] = np.asarray(experience_replay_example["current_state"])
        experience_replay_example["next_state"] = np.asarray((experience_replay_example["next_state"]))

        #append example to memory, update index
        self.experience_replay[self.experience_replay_counter] = experience_replay_example
        self.experience_replay_counter += 1


    def preprocess_observation(self, observation, prediction=False):
        """
        Helper function for preprocessing an observation for consumption by our
        deep learning network
        """
        grayscale_observation = color.rgb2gray(observation)
        resized_observation = transform.resize(grayscale_observation, (1, self.processed_image_dim, self.processed_image_dim)).astype('float32')
        if prediction:
            resized_observation = np.expand_dims(resized_observation, 0)
        return resized_observation


    def take_action(self, observation, log_path, episode_count):
        """
        Given an observation, the model attempts to take an action
        according to its q-function approximation
        """

        #allow model to run prediction, take the max output as the action taken
        prediction = self.model.predict(observation)
        action_taken = np.argmax(prediction)

        #with probability epsilon select a random action
        if random.random() < self.epsilon:
            action_taken = random.randint(0, 7)

        #log q-values (used for various visualizations)
        max_q = prediction[0][action_taken]
        log_str = str(max_q)
        # for q_val in prediction[0]:
        #     log_str += ",%s" % q_val
        # with open(os.path.join(log_path, "%s.log" % episode_count), "a") as f:
        #     f.write(log_str + "\n")

        return action_taken

    def learn(self):
        """
        Allow the model to collect examples from its experience replay memory
        and learn from them
        """

        #if the frame threshold hasn't been reached, iteratively decrease
        #epsilon until it reaches a minimum of 0.1
        if self.cur_frames <= self.max_frames:
            self.epsilon = max(self.epsilon - 1.0/self.max_frames, 0.1)
            # print self.epsilon
            self.cur_frames += 1

        #utilize experience replay to build loss function implemented in paper
        X = []
        y = []

        found_instances = 0
        for i in range(self.minibatch_size):

            #select an experience from memory
            exp_rep_index = random.randint(0, self.min_experiences-1)
            experience = self.experience_replay[exp_rep_index]

            #update model input
            X.append(experience["current_state"])

            #update expected model output (according to loss function described)
            #in paper
            update_val = experience["reward"]
            if not experience["is_terminal"]:
                prediction = self.model.predict(np.expand_dims(experience["next_state"], 0))
                max_q = np.amax(prediction)
                update_val += self.learning_rate * max_q

            cur_q = self.model.predict(np.expand_dims(experience["current_state"], 0))[0]

            #only y-value with a "loss" is the value with the same index as the action
            #taken - the rest of the values are identical to the predicted value,
            #which prevents backpropagation from occurring
            cur_q[experience["action"]] = update_val

            #update expected model output
            y.append(cur_q)

        #convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y).reshape((self.minibatch_size, 8))

        #learn from data
        self.model.train_on_batch(X, y)
