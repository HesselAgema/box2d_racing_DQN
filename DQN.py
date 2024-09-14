import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Flatten the output from the conv layers
        self.flatten = nn.Flatten()

        # Assuming the input size and assuming a square input image, calculate the flattened size
        # We'll use a dummy tensor to calculate the output size of the final conv layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 96, 96)  # Adjust 96x96 based on actual input size
            dummy_output = self.forward_conv(dummy_input)
            flattened_size = dummy_output.shape[1]  * dummy_output.shape[2] * dummy_output.shape[3]

        # Define the fully connected layer
        self.fc = nn.Linear(flattened_size, num_actions)

    def forward_conv(self, x):
        # go through all conv2d layers with relu when calling this function with x (probably preprocessed images)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        # Pass through the conv layers
        x = self.forward_conv(x)
        
        # Flatten the output
        x = self.flatten(x)
        
        # Pass through the fully connected layer to produce the Q-values
        return self.fc(x)
