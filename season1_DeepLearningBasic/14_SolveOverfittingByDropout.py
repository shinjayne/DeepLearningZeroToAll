import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random
from Xavier import xavier_init as xi

mnist = input_data.read_data_sets("MNIST_data/", one_hot =True)

#Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1


#tf Graph Input
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

#Store layers weight&bias
W1= tf.get_variable("W1", shape = [784,256] , initializer = xi(784, 256))
W2= tf.get_variable("W2", shape = [256,256] , initializer = xi(256, 256))
W3= tf.get_variable("W3", shape = [256,256] , initializer = xi(256, 256))
W4= tf.get_variable("W4", shape = [256,256] , initializer = xi(256, 256))
W5= tf.get_variable("W5", shape = [256,256] , initializer = xi(256, 256))
W6= tf.get_variable("W6", shape = [256,256] , initializer = xi(256, 256))
W7= tf.get_variable("W7", shape = [256,256] , initializer = xi(256, 256))
W8= tf.get_variable("W8", shape = [256,10] , initializer = xi(256, 10))

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([256]))
B4 = tf.Variable(tf.random_normal([256]))
B5 = tf.Variable(tf.random_normal([256]))
B6 = tf.Variable(tf.random_normal([256]))
B7 = tf.Variable(tf.random_normal([256]))
B8 = tf.Variable(tf.random_normal([10]))

#Construct Model
dropout_rate = tf.placeholder("float")

_L1 = tf.nn.relu(tf.add(tf.matmul(X,W1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2), B2))
L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2,W3), B3))
L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3,W4), B4))
L4 = tf.nn.dropout(_L4, dropout_rate)
_L5 = tf.nn.relu(tf.add(tf.matmul(L4,W5), B5))
L5 = tf.nn.dropout(_L5, dropout_rate)
_L6 = tf.nn.relu(tf.add(tf.matmul(L5,W6), B6))
L6 = tf.nn.dropout(_L6, dropout_rate)
_L7 = tf.nn.relu(tf.add(tf.matmul(L6,W7), B7))
L7 = tf.nn.dropout(_L7, dropout_rate)
hypothesis = tf.add(tf.matmul(L7,W8),B8)

#Define the cost and Optimizer(learning)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y)) #hypothesis 가 softmax를 취하지 않아도, 이 메서드가 softmax로 했을때의 cost를 계산해줌
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#Initialize Variables
init = tf.initialize_all_variables()

#RUN
with tf.Session() as sess :
    sess.run(init)
    # Training Cycle
    for epoch in range(training_epochs) :
        avg_cost = 0.
        total_batch= int(mnist.train.num_examples/batch_size)
        #Loop over all batches
        for batch in range(total_batch) :
            batch_xs ,batch_ys = mnist.train.next_batch(batch_size)
            #Training in batch
            sess.run(optimizer , feed_dict = {X: batch_xs , Y: batch_ys, dropout_rate : 0.5})
            avg_cost += sess.run(cost , feed_dict = {X: batch_xs, Y : batch_ys, dropout_rate : 0.5}) / batch_size

        if epoch % display_step == 0 :
            print ( "Epoch : ", '%04d' %(epoch+1), "cost =" , "{:.9f}".format(avg_cost) )

    print ("Optimization Finished!")

    #Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
    #Calculate Accuracy
    accuracy  = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print( "Test result : Accuracy = ", accuracy.eval({X:mnist.test.images, Y:mnist.test.labels, dropout_rate : 1}) )








