# config.py
import os
from sympy import false

PHAZE = 'test'

# Training settings
BATCH_SIZE = 1  # Batch size
EPOCH = 40      # Number of training epochs
ENC_LAYER = 5     # Number of encoder layers
HEAD = 8          # Number of attention heads
FEATURE = 16
D_MODEL = 128
DIM = 512  # Embedding dimension
LR = 0.001
MAX_LR = 0.001       # Learning rate
NUM_CLASSES = 2
FF_MULTIPLIER = 4
FF_DIM = 256
DROPOUT = 0.1
IN_CHANNEL = 128
OUT_CHANNEL = 128
WARMUP_EPOCHS = 5
STEP_SIZE = 10
GAMMA = 0.1

# Data processing settings
STEP = 200        # Step size for sampling
SEQ_LENGTH_CNN = 800
SEQ_LENGTH_TRANS = 100

# Dataloader settings
SHUFFLE_FLAG = True     # Whether to shuffle data in dataloader
NUM_WORKERS = 1          # Number of workers for data loading
DROP_LAST = False        # Whether to drop the last incomplete batch

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # The base directory of the project
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoint')  # Path to save/load model checkpoints
BENIGN_DATA_DIR = os.path.join(BASE_DIR, 'benign')
MALICIOUS_DATA_DIR = os.path.join(BASE_DIR, 'malicious')                 # Data directory
TEST_BENIGN_DATA_DIR = os.path.join(BASE_DIR, 'test_benign')
TEST_MALICIOUS_DATA_DIR = os.path.join(BASE_DIR, 'test_malicious')
LOG_FILE = os.path.join(BASE_DIR, 'train.log')
CHECKPOINT_EPOCH  = 40
RETRAIN = False
