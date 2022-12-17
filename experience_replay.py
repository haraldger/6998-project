import sys
import random
import torch

DTYPE = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReplayBuffer:
  def __init__(self,capacity=3):
    self.buffer=[]
    self.capacity=capacity

  def add(self,experience):
    if(self.capacity==len(self.buffer)):
      self.buffer.pop(0)
    self.buffer.append(experience)
  
  def show(self):
    print(".....")
    print("\nCapacity- ", self.capacity  )
    print("\nCurrent length- ",len(self.buffer))
    print(".....")

  def sample(self):
    return(self.buffer[random.randint(0,len(self.buffer))])

  def sample_tensor_batch(self, batch_size):
    state_batch = list()
    action_batch = list()
    next_state_batch = list()
    reward_batch = list()

    for i in range(batch_size):
      experience = self.sample()
      state_batch.append(experience.current_state)
      action_batch.append(experience.action)
      next_state_batch.append(experience.next_state)
      reward_batch.append(experience.reward)

    try:
      state_batch = torch.Tensor(state_batch).to(DEVICE)
    except:
      print(state_batch)
      self.show()
      state_batch = torch.Tensor(state_batch).to(DEVICE)

    action_batch = torch.Tensor(action_batch).to(DEVICE)
    next_state_batch = torch.Tensor(next_state_batch).to(DEVICE)
    reward_batch = torch.Tensor(reward_batch).to(DEVICE)

class Experience:
  def __init__(self, current_state, action, next_state, reward):
    self.current_state = current_state
    self.action = action
    self.next_state = next_state
    self.reward = reward

  

  def show(self):
    print(".....")
    print("\nCurrent State- ", self.current_state  )
    print("\nAction- ",self.action)
    print("\nReward- ",self.reward)
    print("\nNext State",self.next_state)
    print(".....")
