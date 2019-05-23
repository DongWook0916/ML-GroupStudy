# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data
nb_classes=10
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X=tf.placeholder(tf.float32, [None, 784])
Y=tf.placeholder(tf.float32,[None,nb_classes])


W=tf.Variable(tf.random_normal([784,10]))
b=tf.Variable(tf.random_normal([10]))

h= tf.nn.softmax(tf.matmul(X,W)+b)
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(h),axis=1))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction=tf.argmax(h,1)
is_correct=tf.equal(prediction, tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))

num_epochs=15
batch_size=100
num_iterations=int(mnist.train.num_examples/batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(num_epochs):
        avg_cost=0
        for i in range(num_iterations):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _,cost_val=sess.run([optimizer,cost],feed_dict={X:batch_xs, Y:batch_ys})
        print("Epoch: {:04d}, Cost: {:.9f}".format(e + 1, avg_cost))
     
    print("Learning finished")   
    print("Accuracy: ",accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}),)
    
    r=random.randint(0,mnist.test.num_examples-1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print("Prediction: ",sess.run(tf.argmax(h, 1), feed_dict={X: mnist.test.images[r : r + 1]}),)
    plt.imshow(mnist.test.images[r : r + 1].reshape(28, 28),cmap="Greys",interpolation="nearest",)
    plt.show()        