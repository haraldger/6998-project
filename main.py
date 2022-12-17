import sys
import random
import numpy as np
from skimage.transform import resize
import torch
import gymnasium as gym
from swin_agent import SwinAgent
from epsilon_scheduler import EpsilonScheduler
from experience_replay import Experience, ReplayBuffer

# Swin Transformer
sys.path.append('./Swin-Transformer')
from models.swin_transformer_v2 import SwinTransformerV2 as Transformer

# Variables
DTYPE = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPISODES = int(200E6)
REPLAY_MEMORY = 1E6
# INITIAL_EPSILON = 1.0
# FINAL_EPSILON=0.01
# DECAY_FRAMES=1E6
# DECAY_MODE='single'
# DECAY_RATE=0.1

def main():
    ms_pacman()

def ms_pacman():
    # Init data structures
    q_network = get_model()
    target_network = get_model()
    epsilon_scheduler = EpsilonScheduler(decay_frames=40000)
    replay_buffer = ReplayBuffer(capacity=REPLAY_MEMORY)
    agent = SwinAgent(q_network, target_network, epsilon_scheduler, replay_buffer, num_actions=7, initial_exploration=20000)

    # Environment
    env = gym.make('ALE/MsPacman-v5')
    next_state, info = env.reset()
    next_state = process_state(next_state)

    total_reward = 0
    for episode in range(EPISODES):
        flag = False

        previous_state = next_state

        # Act
        action = agent.act(previous_state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = process_state(next_state)

        total_reward += reward

        # Experience replay
        experience = Experience(previous_state, action, next_state, reward)
        replay_buffer.add(experience=experience)

        # Learn
        agent.learn()

        # Update frame counters
        agent.step()
        epsilon_scheduler.step()

        # Environment
        if terminated or truncated:
            print(total_reward)
            total_reward = 0
            next_state, info = env.reset() 
        
    
def process_state(state):
    state = resize(state, (84, 84))
    state = np.moveaxis(state, 2, 0)
    state = np.expand_dims(state, 0)
    return state


def get_model(image_size=(84,84), patch_size=3, in_channels=3,
            num_actions=7, depths=[2,3,2], heads=[3,3,6],
            window_size=7, mlp_ratio=4, drop_path_rate=0.1):
    """
    Default settings are appropriate for Atari games. For other environments, change patch size
    and window size to be compatible, as well as image size to match the environment. Pre-
    processing of the environment may be necessary, preferred format is 256x256, or 84x84 scaled
    by a factor of two (168x168, 336x336, etc.).
    num_actions corresponds to actions available in the environment.
    in_channels should be 1, 3 or 4, for greyscale, RGB, etc.
    """
    return Transformer(img_size=image_size, patch_size=patch_size, in_chans=in_channels,
                    num_classes=num_actions, depths=depths, num_heads=heads, window_size=window_size,
                    mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate)




if __name__ == "__main__":
    main()