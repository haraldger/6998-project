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
                gamma=0.99, alpha=0.0000625, batch_size=32, loss_fn=torch.nn.MSELoss) -> None:
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
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=alpha)
        self.flag=False
        

    def act(self, state):
        state = torch.from_numpy(state).to(DEVICE)
        epsilon = self.epsilon_scheduler.get_epsilon()
        if epsilon > random.random():
            action = random.randrange(self.num_actions)
        else:
            action = torch.max(self.Q(state))
        return action

    def learn(self):
        if self.frames_counter < self.initial_exploration:
            return

        # Sample batch from replay memory
        state_batch, action_batch, next_state_batch, reward_batch = self.replay_buffer.sample_tensor_batch(self.batch_size)
        if(self.flag==False):
            flag=True
            print(state_batch)

        # Compute Bellman loss/update
        q_values = self.Q(state_batch).gather(1, action_batch)
        target_q_values = self.Q_target(next_state_batch).detach()
        targets = reward_batch + self.gamma * target_q_values.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_fn(q_values, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backwards()
        self.optimizer.step()

        if self.frames_counter % self.sync_frequency == 0:
            state_dict = self.Q.state_dict()
            self.Q_target.load_state_dict(state_dict)


    def loss(self, state_batch, action_batch, next_state_batch, reward_batch):
        # TODO
        raise NotImplementedError


    def step(self):
        self.frames_counter += 1



