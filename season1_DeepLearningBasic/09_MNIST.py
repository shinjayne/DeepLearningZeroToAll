import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random


#tf Graph Input
X = tf.placeholder(tf.float32, [None, 784]) #one pixel = 28*28 = 784
Y = tf.placeholder(tf.float32, [None, 10])  # 0~9  Handwriting Recognition = 10 classes

#Set Model Weight
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


#Create Model (Activation, Hypothesis)
activation = tf.nn.softmax(tf.matmul(X, W) + b)

#Cost function
cost = tf.reduce_mean(tf.reduce_sum(-Y*tf.log(activation), reduction_indices = 1))

#Fitting
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


#Initialize variables
init = tf.initialize_all_variables()

training_epoch = 25
display_step = 5
batch_size = 100
mnist = input_data.read_data_sets("MNIST_data/", one_hot =True)

with tf.Session() as sess :
    sess.run(init)

    for epoch in range(training_epoch) :
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        #Loop over all batchs
        for i in range(total_batch) :
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict = {X : batch_xs, Y : batch_ys})
            avg_cost += sess.run(cost, feed_dict = {X : batch_xs, Y : batch_ys} ) / total_batch
        #Show log per epoch step


        if epoch % display_step == 0 :
            print("Epoch : %04d" %(epoch+1) , "cost = ", "{:.9f}".format(avg_cost))
            print(" -> b : " ,sess.run(b))
            print("\n")

    print("Optimization Finished!!\n")

    #Testing the model
    prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(Y ,1))

    accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
    print("\nAccuracy : ", accuracy.eval({X : mnist.test.images, Y : mnist.test.labels}))

    #Get one and predict
    r= random.randint(0, mnist.test.num_examples -1)
    print( "Label:  " , sess.run( tf.argmax(Y, 1) , {Y: mnist.test.labels[r:r+1]} ))
    print( "Prediction : ", sess.run( tf.argmax(activation , 1), {X : mnist.test.images[r:r+1]}))

    #show the img
    plt.imshow(mnist.test.images[r].reshape(28,28), cmap = "Greys", interpolation = "nearest" )
    plt.show()






