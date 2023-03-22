import gym
import numpy as np
import torch
from collections import deque
import random
#from torchmetrics import Accuracy
import torch.nn.functional as F
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

q_network = torch.nn.Sequential(
    torch.nn.Linear(env.observation_space.shape[0], 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,env.action_space.n)
)

loss_fn = torch.nn.MSELoss() 
learning_rate = 0.0001
optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
losses = [] 


target_q_network = torch.nn.Sequential(
    torch.nn.Linear(env.observation_space.shape[0], 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,env.action_space.n)
)

gamma = 0.99
epsilon = 1
epsilon_decay = 0.995
min_epsilon = 0.01


total_epoch = 5000
max_time_steps = 500
replay_memory_size = 10000
replay_memory = deque(maxlen=replay_memory_size)
last_200_score = deque(maxlen=200)
batch_size = 64
sync_networks = 5
tau = 0.003
epsilon_update_count = 0  
average_reward_dqn = 0
best_reward_dqn = 0
scores=[]
total_epsilon_decay = []

for epoch in range(1,total_epoch):
  print("Epoch",epoch)
  score = 0
  step = 0
  state_1 = env.reset()[0]

  for i in range(max_time_steps):
    state_1 = torch.from_numpy(state_1).float().unsqueeze(0)
    q_network.eval()
    with torch.no_grad():
      q_values = q_network(state_1)
    q_network.train()

    if random.random() < epsilon:
      action = random.choice(np.arange(env.action_space.n))
    else:
      action = np.argmax(q_values.data.numpy())
    
    state_2, reward, done, info,_ = env.step(action)
    replay_memory.append([state_1, action, reward, state_2, done])

    step = (step+1)%4
    if(len(replay_memory) > batch_size and step==0):
      minibatch = random.sample(replay_memory,batch_size)
      states = torch.from_numpy(np.vstack([sample[0] for sample in minibatch if sample is not None])).float()
      actions = torch.from_numpy(np.vstack([sample[1] for sample in minibatch if sample is not None])).long()
      rewards = torch.from_numpy(np.vstack([sample[2] for sample in minibatch if sample is not None])).float()
      next_states = torch.from_numpy(np.vstack([sample[3] for sample in minibatch if sample is not None])).float()
      dones = torch.from_numpy(np.vstack([sample[4] for sample in minibatch if sample is not None]).astype(np.uint8)).float()


      ddqn_q_network_values = target_q_network(next_states).detach().max(1)[1]
      q_targets_next = target_q_network(next_states).detach().gather(1, ddqn_q_network_values.unsqueeze(1))
      q_targets = rewards + gamma * q_targets_next * (1 - dones)

      q_expected = q_network(states).gather(1, actions)
      
      loss = F.mse_loss(q_targets,q_expected)
      optimizer.zero_grad()
      losses.append(loss.item())
      loss.backward()
      optimizer.step()


    state_1 = state_2
    score+=reward
    if done:
      break
  
  for target_param, local_param in zip(target_q_network.parameters(), q_network.parameters()):
      target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    #----------------------END of STEP LOOP--------------------

  scores.append(score)
  last_200_score.append(score)
  epsilon = max(min_epsilon, epsilon_decay*epsilon)
  total_epsilon_decay.append(epsilon)
  if np.mean(last_200_score) >= 500:
    print("Environment Solved")
    torch.save(q_network.state_dict(), 'checkpoint_cartpole.pth')
    break


plt.plot(scores)
