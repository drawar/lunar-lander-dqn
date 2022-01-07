import torch
import torch.nn as nn


class DQN(nn.Module):
    """Deep Q Network."""

    def __init__(self, num_states, num_actions, seed):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_states,
                               out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=7 * 7 * 64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=num_actions)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
