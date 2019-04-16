"""
Main file for running deep reinforcement learning simulation

Prabhat Rayapati (pr2sn)
Zack Verham (zdv8rb)
Deep Learning for Computer Graphics (Fall 2016)
Final Project
"""

import gym
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import sgd
from skimage import io, color, transform
import numpy as np
import time
import os
import random

from agent import Agent


#####
# Hyperparameters
#####

GAME_TYPE = 'MsPacman-v0'

#environment parameters
NUM_EPISODES = 100000
MAX_TIMESTEPS = 1000
FRAME_SKIP = 4
PHI_LENGTH = 4

#agent parameters
NAIVE_RANDOM = False
EPSILON = 0.1
EXPERIENCE_REPLAY_CAPACITY = 1000000
MINIBATCH_SIZE = 32
LEARNING_RATE = 0.1
PREPROCESS_IMAGE_DIM = 84

#I/O parameters
OUTPUT_DIR = "./tmp"
LOG_NAME = "output_tmux_final_2_1.log"
MONITOR_DIR = "ms-pacman-experiment-tmux_final_2_1"
EPISODES_DIR = "episode_data_final_2_1"


def verbose_video_callable(index):
    """
    record every iteration
    """
    return True

def hundred_video_callable(index):
    """
    record every 100th iteration
    """
    if index % 100 == 0:
        return True
    return False

def preprocess_observation(observation):
    """
    preprocesses a given observation following the steps described in the paper
    """
    grayscale_observation = color.rgb2gray(observation)
    resized_observation = transform.resize(grayscale_observation, (PREPROCESS_IMAGE_DIM, PREPROCESS_IMAGE_DIM)).astype('float32')
    return resized_observation

def run_simulation():
    """
    Entry-point for running Ms. Pac-man simulation
    """

    #initialize output log
    output_log_path = os.path.join(OUTPUT_DIR, LOG_NAME)
    if os.path.exists(output_log_path):
        os.remove(output_log_path)

    #initialize episode directory
    episode_log_dir_path = os.path.join(OUTPUT_DIR, EPISODES_DIR)
    if not os.path.exists(episode_log_dir_path):
        os.mkdir(episode_log_dir_path)

    #clear episode directory
    for element in os.listdir(episode_log_dir_path):
        element_path = os.path.join(episode_log_dir_path, element)
        os.remove(element_path)

    #initialize environment
    env = gym.make(GAME_TYPE)
    env.monitor.start(os.path.join(OUTPUT_DIR, MONITOR_DIR), force=True, video_callable=hundred_video_callable)
    env.reset()

    #print game parameters
    # print "~~~Environment Parameters~~~"
    # print "Num episodes: %s" % NUM_EPISODES
    # print "Max timesteps: %s" % MAX_TIMESTEPS
    # print "Action space: %s" % env.action_space
    # print
    # print "~~~Agent Parameters~~~"
    # print "Naive Random: %s" % NAIVE_RANDOM
    # print "Epsilon: %s" % EPSILON
    # print "Experience Replay Capacity: %s" % EXPERIENCE_REPLAY_CAPACITY
    # print "Minibatch Size: %s" % MINIBATCH_SIZE
    # print "Learning Rate: %s" % LEARNING_RATE

    #initialize agent
    agent = Agent(epsilon=EPSILON,
                experience_replay_capacity=EXPERIENCE_REPLAY_CAPACITY,
                minibatch_size=MINIBATCH_SIZE,
                learning_rate=LEARNING_RATE)
    agent.load_model("tmp/model/2000.h5")

    #initialize auxiliary data structures
    state_list = [] #captures current phi queue (reference paper for explanation)
    tot_frames = 0

    for i_episode in range(NUM_EPISODES):
        # print "Episode: %s" % i_episode

        #save model for loading later
        if i_episode % 100 == 0 and i_episode != 0:
            agent.save_model(i_episode)

        #initialize instance data structures
        observation = env.reset()
        action = 0
        reward = 0.0
        t = 0
        reward_accum = 0.0

        while True:

            #ensure state list is populated
            if tot_frames < PHI_LENGTH:
                state_list.append(preprocess_observation(observation))
                tot_frames += 1
                continue

            #NOTE: can't render on cs server - need to use the logger to output video
            #env.render()

            #collect current state
            experience_replay_example = {}
            experience_replay_example["is_valid"] = True
            experience_replay_example["current_state"] = state_list

            #if agent should take an action on this frame...
            if t % FRAME_SKIP == 0 or t == 0:

                #collect agent action
                if NAIVE_RANDOM:
                    action = random.randint(0, 7)
                else:
                    # print "taking action..."
                    action = agent.take_action(np.expand_dims(np.asarray(state_list), 0), episode_log_dir_path, i_episode)

                #take agent action
                observation, reward, done, info = env.step(action)

                #update state list with next observation
                state_list.append(preprocess_observation(observation))
                state_list.pop(0)

                #update experience replay instance
                experience_replay_example["action"] = action
                experience_replay_example["reward"] = reward
                experience_replay_example["next_state"] = state_list
                experience_replay_example["is_terminal"] = done

                #add experience to agent memory
                agent.append_experience_replay_example(experience_replay_example)

                #if agent has experienced enough to begin learning,
                #run experience replay operation
                if tot_frames > MINIBATCH_SIZE:
                    agent.learn()

            #otherwise, agent takes the same action it did last frame
            else:
                observation, reward, done, info = env.step(action)

            #update auxiliary hyperparameters
            tot_frames += 1

            #if episode ended, log results and reset
            if done:
                with open(output_log_path, "a") as f:
                    f.write("%s,%s,%s\n" % (i_episode, t+1, reward_accum))
                print("Episode finished after {} timesteps".format(t+1))
                break

            #update instance data structures
            t += 1
            reward_accum += reward

    #clean up
    env.monitor.close()

#entry point
if __name__ == "__main__":
    run_simulation()
