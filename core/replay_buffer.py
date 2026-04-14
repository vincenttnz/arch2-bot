import random


class ReplayBuffer:

    def __init__(self, size=5000):
        self.buffer = []
        self.size = size

    def add(self, state, action, reward):
        self.buffer.append((state, action, reward))
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def sample(self, batch_size=32):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
