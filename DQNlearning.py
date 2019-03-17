import random
import gym
import numpy as np
from collections import deque
import torch.nn as nn
import torch
import time

ENV_NAME = "CartPole-v1"

GAMMA = 0.99
LEARNING_RATE = 0.0005

MEMORY_SIZE = 1000000
BATCH_SIZE = 20
    
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
PATH = 'model.pth'

class Netx(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        hiddenlayers = 64

        self.fc1 = nn.Linear(observation_space, hiddenlayers)
        self.bn1 = nn.BatchNorm1d(hiddenlayers)
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(hiddenlayers, hiddenlayers)
        self.bn2 = nn.BatchNorm1d(hiddenlayers)

        self.fc3 = nn.Linear(hiddenlayers, action_space)
        self.bn3 = nn.BatchNorm1d(action_space)

    def forward(self, x):

        out = self.fc1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.fc3(out)
        # out = self.bn3(out)
        out = self.relu(out)

        return out


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        ## Model and the optimizer
        self.model = Netx(observation_space, action_space)
        self.model.double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        with torch.no_grad():
            q_values = self.model(torch.tensor(state))
            return torch.argmax(q_values[0]).item()

    def save_model(self):
            torch.save(self.model, PATH)     

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                with torch.no_grad():
                    preds = self.model(torch.tensor(state_next))
                    q_update = (reward + GAMMA * torch.max(preds[0]).item())
            q_values_output = self.model(torch.tensor(state))
            q_values_target = q_values_output.clone().detach().requires_grad_(False)
            q_values_target[0][action] = q_update
            loss = self.criterion(q_values_output, q_values_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.exploration_rate *= EXPLORATION_DECAY**2
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def inference():
    model = torch.load(PATH)
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    num_epsoid = 0
    while num_epsoid <1000000:
        num_epsoid += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            time.sleep(0.01)
            env.render()
            with torch.no_grad():
                q_values = model(torch.tensor(state))
                action = torch.argmax(q_values[0]).item()
            #action = DQNSolver(observation_space, action_space).act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            state = state_next
            if terminal:
                print ("Epsoid: " + str(num_epsoid) + "score: " + str(step))
                break


def train():
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    num_epsoid = 0
    l = []
    start = time.time()
    while num_epsoid < 200:
        if len(l)>8:
            break
        num_epsoid += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print ("Epsoid: " + str(num_epsoid) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                if step >=150:
                    l.append(step)
                break
            dqn_solver.experience_replay()
    end = time.time()
    print ("time taken to train: ",end-start)
    dqn_solver.save_model()



if __name__ == "__main__":
    #train()
    #print ("---------------------inferencing now---------------------------")
    inference()