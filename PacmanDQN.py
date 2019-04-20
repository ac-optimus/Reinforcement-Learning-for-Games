''' Code inspired from https://github.com/gsurma/cartpole, ported to pytorch.'''

import random
import gym
import numpy as np
from collections import deque
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
from plotUtils import plot_

ENV_NAME = "MsPacman-ram-v0"
GAMMA = 0.99
LEARNING_RATE = 0.0002

MEMORY_SIZE = 1000000
BATCH_SIZE = 20
    
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
PATH = 'Pacman_model.pth'

HIDDENLAYER_COUNT = 64

class Netx(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        hiddenlayers = HIDDENLAYER_COUNT
        self.fc1 = nn.Linear(observation_space, hiddenlayers)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hiddenlayers, hiddenlayers)
        self.fc3 = nn.Linear(hiddenlayers, action_space)

    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out


class DQNSolver:

    def __init__(self, observation_space, action_space, agent = "DQN"):
        self.exploration_rate = EXPLORATION_MAX
        self.agent = agent
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        ## Model and the optimizer
        self.q_traget = Netx(observation_space, action_space)
        self.q_network = Netx(observation_space, action_space)
        self.optimizer = torch.optim.Adam(self.q_traget.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.flag = 0


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        with torch.no_grad():
            q_values = self.q_traget(torch.tensor(state))
            return torch.argmax(q_values[0]).item()

    def update_q_network(self,u):
        self.q_network.load_state_dict(u)

    def save_model(self):
            torch.save(self.q_traget, PATH)     

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        weight_i_minus_1 = self.q_network.state_dict()
        for state, action, reward, state_next, terminal in batch:
            
            q_update = reward
            if not terminal:
                with torch.no_grad():
                      #  next_state = torch.tensor(state_next).float()
                   # state = torch.tensor(state).float()
                    preds = self.q_network(state_next)
                    update_q = torch.max(preds[0]).item()
                    if self.agent == "DDQN":
                        preds = self.q_traget(state_next)
                        action = torch.argmax(preds[0]).item()
                        pred2 = self.q_network(state_next)
                        update_q = pred2[0][action]
                    q_update = (reward + GAMMA * update_q)
            
            q_values_output = self.q_traget(state)
            q_values_target = q_values_output.clone().detach().requires_grad_(False)
            q_values_target[0][action] = q_update
            loss = self.criterion(q_values_output, q_values_target)
            self.optimizer.zero_grad()
            
            loss.backward()
            self.optimizer.step()
            weight_i = self.q_traget.state_dict()
            #compare_models(self.q_network, self.q_traget)

            if self.flag == 1:#
                #only update qnetwork when one step difference
                self.update_q_network(weight_i_minus_1)
            weight_i_minus_1 = weight_i

            self.flag = 1

        self.exploration_rate *= EXPLORATION_DECAY**2
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Different', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Same model')
        
def train(agent):
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space, agent)
    num_epsoid = 0
    l = []
    start = time.time()
    while num_epsoid < 200:

        num_epsoid += 1
        state = env.reset()
        state = torch.tensor(state).float()
        state = np.reshape(state, [1, observation_space])
        cummulative_reward = 0
        while True:
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            state_next = torch.tensor(state_next).float()
            cummulative_reward +=reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print ("Epsoid: " + str(num_epsoid) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(cummulative_reward))
                break

            dqn_solver.experience_replay()

    end = time.time()
    print ("time taken to train: ",end-start)
    dqn_solver.save_model()

def inference():
    model = torch.load(PATH)
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    num_epsoid = 0
    lis = []
    count =0
    while num_epsoid <1000:
        num_epsoid += 1
        state = env.reset()
        state = torch.tensor(state).float()
        state = np.reshape(state, [1, observation_space])
        step = 0
        cummulative_score = 0
        while True:
            #time.sleep(0.01)
            env.render()
            with torch.no_grad():
                q_values = model(torch.tensor(state))
                action = torch.argmax(q_values[0]).item()
            #action = DQNSolver(observation_space, action_space).act(state)
            state_next, reward, terminal, info = env.step(action)
            cummulative_score += reward
            state_next = torch.tensor(state_next).float()
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            state = state_next
            if terminal:
                lis.append(cummulative_score)
                if step>200:
                    count+=1
                print ("Epsoid: " + str(num_epsoid) + "score: " + str(cummulative_score))
                break
    plot_(np.arange(len(lis)),lis,"No of episode","Reward","Win Rate",200,count)
#    (x_, y_,x_label="Not given", y_label = "Not given", title_ = "Not given", Thresh = None):



if __name__ == "__main__":
    agent  = "DQN"
   # train(agent)
    print ("---------------------inferencing now---------------------------")
    inference()
