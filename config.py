MAX_LENGTH = 20
USE_CUDA = True
SOS_token = 0
EOS_token = 1
MODEL_NAME = 'attn'
SAVE_DIR = './models'
DATA_DIR = './data'
FIG_DIR = './figs'
MODEL_ADD_NAME = ''

# Configuring training
n_epochs = 150000
plot_every = 200
print_every = 1000
learning_rate = 0.0001
hidden_size = 256
n_layers = 2
dropout_p = 0.05
