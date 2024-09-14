import torch
import numpy as np
from DQN import DQN  # Import the DQN network
from ReplayMemory import ReplayMemory
import torch.nn.functional as F
import time



class DQNAgent:
    def __init__(self, action_space, state_shape, device='cpu'):
        self.action_space = action_space
        self.device = torch.device(device)
        self.model = DQN(state_shape[0], action_space.n).to(self.device)
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor
        self.batch_size = 64
        self.memory = ReplayMemory(10000)  # Assuming ReplayMemory is defined
        self.learningRate = 0.0001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate)

    def pick_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()  # Exploration
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.max(1)[1].item()  # Exploitation

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # def replay(self):
    #     if len(self.memory) < self.batch_size:
    #         return
        
    #     # Sample a batch of experiences from memory
    #     states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
    #     # Compute Q-values for current states
    #     q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
    #     # Compute the target Q-values
    #     next_q_values = self.model(next_states).max(1)[0]

    #     dones = dones.float()
    #     target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
    #     # Compute loss
    #     loss = F.mse_loss(q_values, target_q_values.detach())
        
    #     # Backpropagation and optimization
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()



    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Start the overall timing
        total_start_time = time.time()

        # Timing the sampling process
        start_time = time.time()
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        print(f"Sampling batch took {time.time() - start_time:.4f} seconds")

        # Timing the Q-value computation for current states
        start_time = time.time()
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        print(f"Computing Q-values for current states took {time.time() - start_time:.4f} seconds")

        # Timing the target Q-value computation
        start_time = time.time()
        next_q_values = self.model(next_states).max(1)[0]
        dones = dones.float()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        print(f"Computing target Q-values took {time.time() - start_time:.4f} seconds")

        # Timing the loss computation
        start_time = time.time()
        loss = F.mse_loss(q_values, target_q_values.detach())
        print(f"Computing loss took {time.time() - start_time:.4f} seconds")

        # Timing the backpropagation and optimization
        start_time = time.time()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f"Backpropagation and optimization took {time.time() - start_time:.4f} seconds")

        # Total time for replay step
        print(f"Total replay step took {time.time() - total_start_time:.4f} seconds")



    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
