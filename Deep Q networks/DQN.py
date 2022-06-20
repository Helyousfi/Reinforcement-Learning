####################################################################
#   This is an implementation of DQN from scratch using pytorch    #
####################################################################

# Import dependencies
from turtle import forward
import torch
import torch.nn as nn
import gym
from collections import deque
import itertools
import numpy as np
import random

# Create hyper params
GAMMA = 0.99 # Discount rate
BATCH_SIZE = 12 
BUFFER_SIZE = 50000 # maximum number of transitions
MIN_REPLAY_SIZE = 1000
EPSILON_END = 0.02
EPSILON_START = 1
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000 # we set target params equal to online

# create the environment
env = gym.make("CartPole-v0")
print(env)
replay_buffer = deque(maxlen=BUFFER_SIZE) # contains states, ...
reward_buffer = deque([0.0], maxlen=100) # ?

episode_reward = 0.0

# Create the network
class Net(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape))
        self.model = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, x):
        return self.model(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0)) # apply the model
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action

# Online and target network
online_net = Net(env)
target_net = Net(env)

# Optimizer
optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

target_net.load_state_dict(online_net.state_dict())

# initialize replay buffer
obs = env.reset()
print(obs)
for _ in range(BUFFER_SIZE):
    action = env.action_space.sample()
    new_obs, rew, done, _  = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs
    if done:
        obs = env.reset()

# Main training loop
obs = env.reset()
for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    # EPSILON GREEDY
    rnd_sample = random.random()
    if  rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs, rew, done, _  = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)

    # update the episode reward
    episode_reward += rew
    if done:
        obs = env.reset()
        reward_buffer.append(episode_reward)
        episode_reward = 0.0
    
    # Start gradient step
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rew = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])


    obses_t = torch.as_tensor(obses, dtype = torch.float32)     
    actions_t = torch.as_tensor(actions, dtype = torch.int64).unsqueeze(-1)
    rew_t = torch.as_tensor(rew, dtype = torch.float32)   
    dones_t = torch.as_tensor(dones, dtype = torch.float32)     
    new_obses_t = torch.as_tensor(new_obses, dtype = torch.float32)     


    # Compute targets
    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rew_t + GAMMA * (1 - dones_t) * max_target_q_values


    #Compute loss
    q_values = online_net(obses_t)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)
    

    #Gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % 1000 == 0:
        print()
        print('step: ', step)
        print('AVG Rew: ', np.mean(reward_buffer))
