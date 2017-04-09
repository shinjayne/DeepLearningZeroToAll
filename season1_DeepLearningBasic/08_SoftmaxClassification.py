import numpy as np
import tensorflow as tf

#Read datafile
xy = np.loadtxt('train.txt', unpack = True, dtype = 'float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

print("x :", x_data  )
print("y :", y_data )

#tf Graph Input
X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 3])



#Set Model Weight
W = tf.Variable(tf.zeros([3,3]))

#Construct Model
hypothesis = tf.nn.softmax(tf.matmul(X,W))  #softmax를 통해 확률로 변경

#Minimize Using cross entropy
learning_rate = 0.01
#Cross entropy
cost = tf.reduce_mean(tf.reduce_sum(-Y*tf.log(hypothesis), reduction_indices= 1 ))  # 모든 testcase 의 평균sum( 한 testcase에서 [Ha, Hb , Hc] & Y: [1, 0 ,0] 에대한 cost의 합 )

#Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Initialize the Variables
init  = tf.initialize_all_variables()

#RUN
with tf.Session() as sess :
    sess.run(init)

    for step in range(4001) :
        sess.run(optimizer, feed_dict = {X :x_data, Y :y_data })
        if step % 200 == 0 :
            print(step, "cost=", sess.run(cost, feed_dict ={X :x_data, Y :y_data }) ,"\n",sess.run(W))
    a =sess.run(hypothesis , feed_dict = {X : [[1, 3, 7]]})
    print("test : " , a, sess.run(tf.arg_max(a,1)))
    b =sess.run(hypothesis , feed_dict = {X : [[1, 7, 5]]})
    print("test : " , b, sess.run(tf.arg_max(b,1)))
    c =sess.run(hypothesis , feed_dict = {X : [[1, 2, 3]]})
    print("test : " ,c, sess.run(tf.arg_max(c,1)))

