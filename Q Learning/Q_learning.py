# Import depedencies
from os import stat
import numpy as np
import gym
import random
import time
# Construct the environment
env = gym.make("FrozenLake-v1")
# Construct the Q matrix
state_space_size = env.observation_space.n 
action_space_size = env.action_space.n 
Q_values = np.zeros((state_space_size, action_space_size))
print(Q_values)
# Define some params
episodes_number = 10000
max_steps_per_episode = 100 # Max steps per episodes if the game doesn't terminate, stop
learning_rate = 0.1
discount_rate = 0.99 # Gamma
# Exploration-Exploitation trade-off
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001
# Q algorithm
rewards_all_episodes = []
for episode in range(episodes_number):
    state = env.reset()
    done = False
    current_episode_rewards = 0.0
    for step in range(max_steps_per_episode):
        # Exploration-Exploitation trade-off 
        exploration_rate_thresh = random.uniform(0, 1)
        if exploration_rate > exploration_rate_thresh:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_values[state,:]) 
        new_state, reward, done, _ = env.step(action)
        # Update Q value
        Q_values[state, action] = (1 - learning_rate) * Q_values[state, action] + \
            learning_rate * (reward + discount_rate * np.max(Q_values[new_state, :]))
        state = new_state
        current_episode_rewards += reward
        if done == True:
            break    
    exploration_rate = min_exploration_rate + (max_exploration_rate-min_exploration_rate) \
        * np.exp(-exploration_decay_rate * episode)
    rewards_all_episodes.append(current_episode_rewards)
reward_per_thousand_episodes = np.split(np.array(rewards_all_episodes), episodes_number/1000)
count = 1000
print("average reward per thousand episodes")
for r in reward_per_thousand_episodes:
    print(count,':', str(sum(r/1000)))
    count += 1000
# print update Q-table
print("Q-table")
print(Q_values)
# Playing frozen Lake Game
for episode in range(4):
    state = env.reset()
    done = False
    time.sleep(1)
    for _ in range(max_steps_per_episode):
        env.render()
        time.sleep(0.3)
        action = np.argmax(Q_values[state, :])
        new_state, rew, done, info = env.step(action)
        if done:
            env.render()
            if rew == 1:
                print("You won!")
                time.sleep(3)
            else:
                print('You fel into a hole')
                time.sleep(3)
            break
        state = new_state
