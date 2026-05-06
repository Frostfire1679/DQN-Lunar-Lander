# -*- coding: utf-8 -*-
"""a.10_exploring_ai.py

Conner Hennessee

Original file is located at
    https://colab.research.google.com/drive/1yOaAek9cuBoFR6BLS1NGqvve5y0Fg_QA
"""

import gymnasium as gym

# Install necessary packages for Lunar Lander and other environments
!pip install gymnasium[box2d] tqdm

print("Packages installed. Creating Lunar Lander environment...")

# Create the Lunar Lander environment
env = gym.make("LunarLander-v3")

# Print some basic information about the environment
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Close the environment after checking
env.close()

print("Lunar Lander environment created successfully!")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the Q-Network architecture
class DQN(nn.Module):
    def __init__(self, obs_space_dims, action_space_dims):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(obs_space_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

print("DQN network architecture defined.")

import random
from collections import deque

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample experiences from the buffer
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.int64).unsqueeze(1),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

print("ReplayBuffer class defined.")

import random
import numpy as np

# Define the DQNAgent
class DQNAgent:
    def __init__(self, obs_space_dims, action_space_dims, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay, replay_buffer_capacity, batch_size):
        self.obs_space_dims = obs_space_dims
        self.action_space_dims = action_space_dims
        self.learning_rate = learning_rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon_start # Exploration-exploitation trade-off
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(obs_space_dims, action_space_dims).to(self.device)
        self.target_net = DQN(obs_space_dims, action_space_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is not trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(replay_buffer_capacity)

    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randrange(self.action_space_dims)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor);
                return q_values.argmax().item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return # Not enough samples to optimize

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # --- Debugging: Print shapes of tensors --- REMOVED
        # print(f"Shape of states: {states.shape}")
        # print(f"Shape of actions: {actions.shape}")
        # print(f"Shape of rewards: {rewards.shape}")
        # print(f"Shape of next_states: {next_states.shape}")
        # print(f"Shape of dones: {dones.shape}")
        # ----------------------------------------

        # Compute Q(s_t, a) - the model predicts Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(states).gather(1, actions)

        # Compute V(s_{t+1}) for all next states.
        # This is where the target network is used.
        # For terminal states, V(s) is 0.
        next_state_values = torch.zeros(self.batch_size, 1, device=self.device)
        with torch.no_grad():
            # FIX: Squeeze the dones tensor to make the mask 1-dimensional for correct indexing
            non_final_mask = ~dones.squeeze().bool()
            non_final_next_states = next_states[non_final_mask]
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].unsqueeze(1)

        # Compute the expected Q values
        expected_state_action_values = rewards + (self.gamma * next_state_values)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1) # Clip gradients to prevent exploding gradients
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

print("DQNAgent class defined.")

# Define hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
REPLAY_BUFFER_CAPACITY = 100000
BATCH_SIZE = 64
NUM_EPISODES = 500
TARGET_UPDATE_FREQ = 10

# Initialize the environment and agent
env = gym.make("LunarLander-v3")
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.n

agent = DQNAgent(
    obs_space_dims=obs_space_dims,
    action_space_dims=action_space_dims,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY,
    replay_buffer_capacity=REPLAY_BUFFER_CAPACITY,
    batch_size=BATCH_SIZE
)

rewards_per_episode = []

print("Starting training loop...")

from tqdm import tqdm

for episode in tqdm(range(NUM_EPISODES), desc="Training Episodes"):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    truncated = False # Add truncated flag

    while not done and not truncated:
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        agent.optimize_model()

    rewards_per_episode.append(episode_reward)

    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

env.close()
print("Training finished!")

"""### Training Results

Let's visualize the rewards per episode to see how the agent's performance improved over time.
"""

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode")
plt.grid(True)
plt.show()