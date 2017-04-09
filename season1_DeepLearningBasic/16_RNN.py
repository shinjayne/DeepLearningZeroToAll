#%%
import tensorflow as tf
import numpy as np

#%%
idx2char = ['h','i','e','l','o']

#Teach : hihell -> ihello
x_data = [[0,1,0,2,3,3]]  #hihell
x_one_hot = [[[1,0,0,0,0],
              [0,1,0,0,0],
              [0,0,1,0,0],
              [0,0,0,1,0],
              [0,0,0,0,1]]]

y_data = [[1,0,2,3,3,4]] #ihello


#%%
num_classes = 5
input_dim = 5 #one-hot size
hidden_size = 5 #output from the LSTM
batch_size = 1 #one sentence
sequence_length = 6

#%%
X = tf.placeholder(
    tf.float32, [None, squence_length, input_dim]) # X one_hot

Y = tf.placeholder(
    tf.float32, [None, sequence_length]) # Y_data

cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)

initial_state - cell.zero_state(batch_size, tf.float32)

outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state = initial_state, dtype=tf.float32
)
