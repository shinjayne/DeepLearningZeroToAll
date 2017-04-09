import tensorflow as tf
import numpy as np

# Load Data
xy = np.loadtxt("xor.txt", unpack = True)
x = np.transpose(xy[0:-1])
y = np.transpose(xy[-1:])

# Make Placeholders
X = tf.placeholder(tf.float32, name = "X_input")
Y = tf.placeholder(tf.float32, name = "Y_input")

W1 = tf.Variable(tf.random_uniform([3,3], -1.0 ,1.0), name = "Weight1")
W2 = tf.Variable(tf.random_uniform([3,1], -1.0 ,1.0), name  = "Weight2")

#Create  Model in NeuralNetwork version
with tf.name_scope("layer2") as scope :
    L2 = tf.sigmoid(tf.matmul(X,W1))

with tf.name_scope("layer3") as scope :
    hypothesis = tf.sigmoid(tf.matmul(L2, W2) )

#Cost
with tf.name_scope("cost") as scope :
    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
    tf.scalar_summary("cost", cost)

#trianing
with tf.name_scope("train") as scope :
    learning_rate = tf.Variable(0.5)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

tf.histogram_summary("weights1", W1)
tf.histogram_summary("weights2", W2)

tf.histogram_summary("y", Y)

init = tf.initialize_all_variables()


with tf.Session() as sess :
    sess.run(init)

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs/xor_logs3" ,sess.graph)

    for step in range(2001) :
        sess.run(optimizer , feed_dict = {X:x , Y:y})
        if step % 200  == 0 :
            summary = sess.run(merged, feed_dict = {X:x ,Y:y})
            writer.add_summary(summary, step)
            print( step ,"cost=" ,sess.run(cost,feed_dict = {X:x , Y:y} ),"\nW1\n", sess.run(W1) ,"\n")


    #Test Model
    testing = tf.equal(tf.floor(hypothesis +0.5) ,Y)

    #Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(testing, "float"))
    accuracy_summ = tf.scalar_summary("accuracy", accuracy)
    asr = sess.run(accuracy_summ, feed_dict = {X:x, Y:y})
    writer.add_summary(asr, step)

    print( sess.run([hypothesis, tf.floor(hypothesis+0.5), testing, accuracy] ,feed_dict = {X:x , Y:y}))
    print( "Accuracy :: " , accuracy.eval({X :x, Y:y}))