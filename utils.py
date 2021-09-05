import math
import time
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from config import *

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def split_dataset():
    ratio = 0.7
    filename = './data/cn-eng.txt'
    train_filename = './data/cn-eng_train.txt'
    valid_filename = './data/cn-eng_valid.txt'
    with open(filename, 'r') as f_all:
        dataset = f_all.read().split('\n')
        random.shuffle(dataset)
        train_data = dataset[:int(ratio * len(dataset))]
        valid_data = dataset[int(ratio * len(dataset)):]
    with open(train_filename, 'w') as f_train:
        for line in train_data:
            if len(line.strip())==0:
                continue
            f_train.write(line + '\n')
    with open(valid_filename, 'w') as f_valid:
        for line in valid_data:
            if len(line.strip())==0:
                continue
            f_valid.write(line + '\n')

# split_dataset()
def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    points = np.save('./figs/loss_{}.npy'.format(MODEL_NAME), np.array(points))
    plt.savefig(os.path.join(FIG_DIR, 'loss_{}.png'.format(MODEL_NAME)))

# split_dataset()