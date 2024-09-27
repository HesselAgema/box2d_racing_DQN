import torch
import numpy as np
from DQN import DQN
from ReplayMemory import ReplayMemory
import torch.nn.functional as F
import os

class DQNAgent:
    def __init__(self, action_space, state_shape, epsilon, epsilon_min, epsilon_decay, 
                 gamma, batch_size, memory_capacity, learning_rate, 
                 target_update_frequency, model_directory, model_filename, device='cpu'):
        self.action_space = action_space
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize DQN models
        self.model = DQN(state_shape[0], action_space.n).to(self.device)
        self.target_model = DQN(state_shape[0], action_space.n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # Copy weights from main model
        self.target_model.eval()  # Set the target model to evaluation mode
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_capacity)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.target_update_frequency = target_update_frequency
        self.model_directory = model_directory
        self.model_filename = model_filename
        self.steps = 0

    def pick_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()  # Exploration
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.max(1)[1].item()  # Exploitation

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device)
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones.float()))
        
        loss = F.mse_loss(q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, episode, rewards_from_episodes):
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        filepath = os.path.join(self.model_directory, f"{self.model_filename}_episode_{episode}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'rewards_from_episodes': rewards_from_episodes  # Save rewards
        }, filepath)
        print(f"Model saved at {filepath}")

    def load_model(self):
        model_files = [f for f in os.listdir(self.model_directory) if self.model_filename in f]
        if not model_files:
            print("No saved models found.")
            return 0, []
        
        latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        filepath = os.path.join(self.model_directory, latest_model)
        
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        episode = checkpoint['episode']
        
        # Load rewards from episodes if available
        rewards_from_episodes = checkpoint.get('rewards_from_episodes', [])
        
        print(f"Loaded model from {filepath}, starting from episode {episode}")
        return episode, rewards_from_episodes
