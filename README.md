DEPENDENCIES
```
pip install gymnasium[all]
pip install gymnasium[accept-rom-license]
pip install gym-tetris
pip install timm
pip install scikit-image
```

PRE-REQUISITES
1. Clone this project
```
git clone https://github.com/haraldger/6998-project
```
2. Run the following command to move into the project:
```
cd 6998-project
```
3. Run the following commands to see that directories exist:
```
ls data/
ls model_weights/
```
If not, run the following commands to create the directories:
```
mkdir data/
mkdir model_weights/
```


TRAIN AGENT FROM SCRATCH

1. Choose which hyperparameters to train with. Some default configurations are provided in different branches. "Main" holds the standard (medium) settings, the branch "high-variables" has higher settings, "low-variables" has low, and "super-high" has the largest possible values for the environment we ran these experiments on - Google Cloud Compute Engine using 2 V100 GPUs, with 8 n1-Standard CPUs.

2. Alternatively, edit the pacman.py script and set the global variables yourself. These are located at the beginning of the file, directly after the import statements. It is not recommended to change anything other than these global variables (they are in capital letters, some are commented and may safely be uncommented).

3. Run the script pacman.py
```
python pacman.py
```

EVALUATE AGENT

1. No script has been provided that easily evaluates the performance of an agent. However, the scripts in the repository automatically saves the entire model (parameter weights, state_dicts, optimizers, etc.) every 10k frames. These are saved to the directory model_weights/, and can be loaded or downloaded from there to be run in your own evaluation script. Resuming training is also possible by loading these state dicts.

2. The scripts automatically saves a graph at the end of each game of Pacman. This graph plots the total number of training frames versus total score achieved in that game. Note: The x-axis plots total frames, over all games. The y-axis plots the cumulative reward of a single game, which finished at the corresponding frame number. This graph can be downloaded to visualize agent performance. Tip: if training is run on tmux or equivalent, you can manually download the graph to see how training has progressed while the tmux session is running, without interrupting the training procedure.



CREDIT
1. arXiv:2206.15269 - https://doi.org/10.48550/arXiv.2206.15269 - Deep Reinforcement Learning with Swin Transformer
2. @misc{gym-tetris,
  author = {Christian Kauten},
  howpublished = {GitHub},
  title = {{Tetris (NES)} for {OpenAI Gym}},
  URL = {https://github.com/Kautenja/gym-tetris},
  year = {2019},} - UNUSED
