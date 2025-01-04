FILE_PATH = "p1_navigation/Banana_Windows_x86_64/Banana.exe"

EPOCHS = 2
EPISODE_LEN = 100

ALPHA = 1

BATCH_SIZE = 32


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network