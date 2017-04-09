import tensorflow as tf
import matplotlib.pyplot as plt

X = [1. , 2., 3.]
Y = [1., 2., 3.]

m= n_samples =len(X)

#Set model weights
W = tf.placeholder(tf.float32)

#Construct a linear model
hypothesis = tf.mul(X,W)

#Cost function
cost = tf.reduce_mean(tf.square(hypothesis-Y ))

#Initializing the variables
init = tf.initialize_all_variables()

#for Graphs
W_val = []
cost_val = []

#Launch the graph (RUN)
sess = tf.Session()
sess.run(init)
for i in range(-30,50) :
    print (i*0.1, sess.run(cost, feed_dict = {W: i*0.1}))
    W_val.append(i*0.1)
    cost_val.append(sess.run(cost, feed_dict = {W: i*0.1}))

#Graphic display
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()