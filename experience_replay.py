import sys
import numpy as np
import random
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReplayBuffer:
    def __init__(self, capacity=40000, dims=(3,84,84)):
        self.capacity = capacity
        self.counter = 0
        self.dims=dims

        self.state_memory = torch.empty(size=(self.capacity, self.dims[0], self.dims[1], self.dims[2])).type(torch.float)
        self.action_memory = torch.empty(size=(self.capacity,1)).type(torch.long)
        self.next_state_memory = torch.empty(size=(self.capacity, self.dims[0], self.dims[1], self.dims[2])).type(torch.float)
        self.reward_memory = torch.empty(size=(self.capacity,1)).type(torch.float)


    def add(self, state, action, next_state, reward):
        idx = int(self.counter % self.capacity)
        self.state_memory[idx] = torch.from_numpy(state).type(torch.float)
        self.action_memory[idx] = torch.Tensor([action]).type(torch.long)
        self.next_state_memory[idx] = torch.from_numpy(next_state).type(torch.float)
        self.reward_memory[idx] = torch.Tensor([reward]).type(torch.float)
        self.counter += 1
    

    def sample_tensor_batch(self, batch_size):
        sample_index = np.random.choice(self.memory_size, batch_size)

        state_sample = torch.FloatTensor((batch_size, self.dims[0], self.dims[1], self.dims[2])).to(DEVICE)
        action_sample = torch.LongTensor((batch_size, 1)).to(DEVICE)
        next_state_sample = torch.FloatTensor((batch_size, self.dims[0], self.dims[1], self.dims[2])).to(DEVICE)
        reward_sample = torch.FloatTensor((batch_size, 1)).to(DEVICE)

        for index in range(sample_index.size):
            state_sample[index] = self.state_memory[sample_index[index]]  
            action_sample[index] = self.action_memory[sample_index[index]]
            next_state_sample[index] = self.next_state_memory[sample_index[index]]
            reward_sample[index] = self.reward_memory[sample_index[index]]

        return state_sample, action_sample, next_state_sample, reward_sample

    def show(self):
        print(".....")
        print("\nCapacity- ", self.capacity)
        print("\nCurrent length- ",len(self.state_memory))
        print(".....")