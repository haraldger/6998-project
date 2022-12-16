import random

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

class Experience:
  def __init__(self,current_state,next_state,action,reward):
    self.current_state= current_state
    self.next_state=next_state
    self.action=action
    self.reward=reward

  def show(self):
    print(".....")
    print("\nCurrent State- ", self.current_state  )
    print("\nAction- ",self.action)
    print("\nReward- ",self.reward)
    print("\nNext State",self.next_state)
    print(".....")
