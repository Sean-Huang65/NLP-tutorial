MAX_LENGTH = 10
USE_CUDA = True
SOS_token = 0
EOS_token = 1
MODEL_NAME = 'gru'
SAVE_DIR = './models'
DATA_DIR = './data'
FIG_DIR = './figs'

# Configuring training
n_epochs = 100000
plot_every = 200
print_every = 1000
learning_rate = 0.0001
hidden_size = 500
n_layers = 2
dropout_p = 0.05
