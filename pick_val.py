import numpy as np
import random

data_in = np.load('./data/Data_in.npy')
data_out = np.load('./data/Data_out.npy')

x = random.sample(range(0, 66), 10)
y = np.ones(66)

for i in range(len(x)):
    y[x[i]] = 0
y = np.nonzero(y)

img_train = data_in[y]
tar_train = data_out[y]
img_val = data_in[x]
tar_val = data_out[x]

np.save('./data/img_train.npy', img_train)
np.save('./data/img_val.npy', img_val)
np.save('./data/tar_train.npy', tar_train)
np.save('./data/tar_val.npy', tar_val)