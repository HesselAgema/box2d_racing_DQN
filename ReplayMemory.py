from collections import deque
import random
import torch
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.stack(state), dtype=torch.float32),
            torch.tensor(action),
            torch.tensor(reward),
            torch.tensor(np.stack(next_state), dtype=torch.float32),
            torch.tensor(done)
        )

    def __len__(self):
        return len(self.memory)
