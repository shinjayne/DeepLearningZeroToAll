import tensorflow as tf

#training data
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1,0 , 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#Our hypoyhesis
hypothesis = W* X + b

#simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#minimize the cost

a= tf.Variable((0.1))
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#Before Starting, initialize the variables. We will run this first.
init = tf.initialize_all_variables()

#Launch the graph
sess = tf.Session()
sess.run(init)

#Fit the line.
for step in range(2001) :
    sess.run(train, feed_dict = {X : x_data, Y : y_data})
    if step % 20 == 0 :
        print ( step, sess.run(cost, feed_dict = {X : x_data, Y : y_data}) , sess.run(W), sess.run(b) )