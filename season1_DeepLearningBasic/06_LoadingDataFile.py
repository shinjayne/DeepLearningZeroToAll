import tensorflow as tf
import numpy as np
#http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numby.loadtxt.html
xy = np.loadtxt('train.txt', unpack = True, dtype = 'float32')
x_data = xy[0:-1]
y_data = xy[-1]

print("x", x_data)
print("y", y_data)


W= tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))


#Hyphotesis
hypothesis = tf.matmul(W , x_data)

#simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#Minimize
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

##initialize the Variables
init = tf.initialize_all_variables()

#Run
sess = tf.Session()
sess.run(init)

#Fit the line
for step in range(2001) :
    sess.run(train)
    if step % 20 == 0 :
        print (step, sess.run(cost), sess.run(W))