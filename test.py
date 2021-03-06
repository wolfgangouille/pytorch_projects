import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(20, 10, True)
      self.fc2 = nn.Linear(10, 3, True)
    def forward(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      return x


# Equates to one random 28x28 image

my_nn = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(my_nn.parameters(), lr=0.01)

input = torch.randn((5,20))
target = torch.randn((5,3))


for i in range(100):
    optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
    output=my_nn(input)
    loss=criterion(output, target)
    loss.backward()
    optimizer.step()
    print(loss)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
