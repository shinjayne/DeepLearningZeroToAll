import tensorflow as tf
import numpy as np


xy = np.loadtxt("xor.txt", unpack = True)
x = np.transpose(xy[0:-1])
y = np.transpose(xy[-1:])


print("x :", x  )
print("y :", y )

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([3,3], -1.0 ,1.0))
W2 = tf.Variable(tf.random_uniform([3,1], -1.0 ,1.0))

#Create  Model in NeuralNetwork version
L2 = tf.sigmoid(tf.matmul(X,W1))
hypothesis = tf.sigmoid(tf.matmul(L2, W2) )

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

#Learning
learning_rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()


with tf.Session() as sess :
    sess.run(init)

    for step in range(2001) :
        sess.run(optimizer , feed_dict = {X:x , Y:y})
        if step % 200 == 0 :
            print( step ,"cost=" ,sess.run(cost,feed_dict = {X:x , Y:y} ),"\nW1\n", sess.run(W1) ,"\n")


    #Test Model
    testing = tf.equal(tf.floor(hypothesis +0.5) ,Y)

    #Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(testing, "float"))
    print( sess.run([hypothesis, tf.floor(hypothesis+0.5), testing, accuracy] ,feed_dict = {X:x , Y:y}))
    print( "Accuracy :: " , accuracy.eval({X :x, Y:y}))