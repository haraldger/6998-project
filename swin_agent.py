import numpy
import random
import torch

from epsilon_scheduler import EpsilonScheduler

# Variables
DTYPE = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SwinAgent:
    def __init__(self, Q_network, target_network, epsilon_scheduler, replay_buffer, num_actions,
                initial_exploration=40000, acting_frequency=4, learning_frequency=4, sync_frequency=40000, 
                gamma=0.99, alpha=0.0000625) -> None:
        self.Q = Q_network.to(device=DEVICE)
        self.Q_target = target_network.to(device=DEVICE)
        self.epsilon_scheduler = epsilon_scheduler
        self.replay_buffer = replay_buffer
        self.num_actions = num_actions

        self.initial_exploration = initial_exploration
        self.acting_frequency = acting_frequency
        self.learning_frequency = learning_frequency
        self.sync_frequency = sync_frequency
        self.frames_counter = 0

        self.gamma = gamma
        self.alpha = alpha
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=alpha)
        

    def act(self, state):
        if self.frames_counter % self.acting_frequency != 0:
            return None 

        epsilon = self.epsilon_scheduler.get_epsilon()
        if epsilon > random.random():
            action = random.randrange(self.num_actions)
        else:
            action = torch.max(self.Q(state))
        return action

    def learn(self):
        if self.frames_counter < self.initial_exploration:
            return
        # if self.frames_counter % self.learning_frequency != 0:
        #     return

        ### TODO: implement batch learning
        """
        1. Sample batch of experiences from replay buffer
        2. Forward call, compute loss - https://arxiv.org/pdf/2206.15269.pdf
        3. Set optimizer zero_grad, call backwards
        """

        if self.frames_counter % self.sync_frequency == 0:
            state_dict = self.Q.state_dict()
            self.Q_target.load_state_dict(state_dict)


    def loss(state):
        # TODO
        raise NotImplementedError


    def step(self):
        self.frames_counter += 1



