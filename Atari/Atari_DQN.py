# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import gym
from gym import spaces
import gym
import numpy as np
import torch
from collections import deque
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gym.wrappers import AtariPreprocessing, FrameStack
from ale_py import ALEInterface
from ale_py.roms import Breakout
import csv
import pandas as pd


def write_to_csv(Even_list):
     with open('/Users/kaushikkumartrs/Desktop/UB_CSE_Masters/Semester_1/Reinforcement_Learning/Project/DQN_Scores_atari.csv', 'w', newline='') as csvfile:
         writer = csv.writer(csvfile)
         writer.writerows(Even_list)
         
         
         
#Environment
ale = ALEInterface()
ale.loadROM(Breakout)
env = gym.make("BreakoutNoFrameskip-v4")
env = AtariPreprocessing(env)
env = FrameStack(env, num_stack=4)

q_network = torch.nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 4))

target_q_network = torch.nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 4))

loss_fn = torch.nn.HuberLoss() 
learning_rate = 0.0001
optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
losses = [] 

q_network.load_state_dict(torch.load('/Users/kaushikkumartrs/Desktop/UB_CSE_Masters/Semester_1/Reinforcement_Learning/Project/q_network_dqn_atari.pth'))
target_q_network.load_state_dict(torch.load('/Users/kaushikkumartrs/Desktop/UB_CSE_Masters/Semester_1/Reinforcement_Learning/Project/target_q_network_dqn_atari.pth'))



gamma = 0.99
epsilon = 1
epsilon_decay = 0.995
min_epsilon = 0.01


total_epoch = 2000
max_time_steps = 10000
replay_memory_size = 100000
replay_memory = deque(maxlen=replay_memory_size)
last_50_score = deque(maxlen=50)
batch_size = 32
sync_networks = 5
tau = 0.003
epsilon_update_count = 0  
average_reward_dqn = 0
best_reward_dqn = 0
scores_dqn=[]
rolling_10_average = []
total_epsilon_decay = []
frame_count=0

for epoch in range(1,total_epoch):
  score = 0
  step = 0
  episode_length =0
  state_1 = env.reset()
  state_1 = state_1[0].__array__() if isinstance(state_1, tuple) else state_1.__array__()
  state_1 = torch.from_numpy(state_1).float().unsqueeze(0)

  for i in range(max_time_steps):
    frame_count+=1
    episode_length+=1
    #q_network.eval()
    with torch.no_grad():
      q_values = q_network(state_1)
    q_network.train()

    if random.random() < epsilon:
      action = random.choice(np.arange(env.action_space.n))
    else:
      action = np.argmax(q_values.data.numpy())
    
    state_2, reward, done, info,_ = env.step(action)
    state_2 = state_2[0].__array__() if isinstance(state_2, tuple) else state_2.__array__()
    state_2 = torch.from_numpy(state_2).float().unsqueeze(0)

    replay_memory.append([state_1, action, reward, state_2, done])

    step = (step+1)%5
    if(len(replay_memory) > batch_size and step==0):
      minibatch = random.sample(replay_memory,batch_size)
      states = torch.from_numpy(np.vstack([sample[0] for sample in minibatch if sample is not None])).float()
      actions = torch.from_numpy(np.vstack([sample[1] for sample in minibatch if sample is not None])).long()
      rewards = torch.from_numpy(np.vstack([sample[2] for sample in minibatch if sample is not None])).float()
      next_states = torch.from_numpy(np.vstack([sample[3] for sample in minibatch if sample is not None])).float()
      dones = torch.from_numpy(np.vstack([sample[4] for sample in minibatch if sample is not None]).astype(np.uint8)).float()


      q_targets_next = target_q_network(next_states).detach().max(1)[0].unsqueeze(1)
      q_targets = rewards + gamma * q_targets_next * (1 - dones)

      q_expected = q_network(states).gather(1, actions)
      
      loss = F.mse_loss(q_targets,q_expected)
      optimizer.zero_grad()
      losses.append(loss.item())
      loss.backward()
      optimizer.step()

    if frame_count%10000 ==0:
      for target_param, local_param in zip(target_q_network.parameters(), q_network.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        target_param.data.copy_(local_param.data)

    state_1 = state_2
    score+=reward
    if done:
      break
    #----------------------END of STEP LOOP--------------------

  scores_dqn.append(score)
  
  last_50_score.append(score)
  epsilon = max(min_epsilon, epsilon_decay*epsilon)
  total_epsilon_decay.append(epsilon)
  print(f"Episode : {epoch}, score = {score}, length = {episode_length}")
  if epoch%10==0:
    write_to_csv((map(lambda x: [x], scores_dqn)))
    torch.save(q_network.state_dict(), '/Users/kaushikkumartrs/Desktop/UB_CSE_Masters/Semester_1/Reinforcement_Learning/Project/q_network_dqn_atari.pth')
    torch.save(target_q_network.state_dict(), '/Users/kaushikkumartrs/Desktop/UB_CSE_Masters/Semester_1/Reinforcement_Learning/Project/target_q_network_dqn_atari.pth')
  if epoch %100==0:
      plt.plot(scores_dqn)
      plt.plot(pd.Series(scores_dqn).rolling(7).mean())
  if np.mean(last_50_score) >= 40:
    print("Environment Solved")
    break

