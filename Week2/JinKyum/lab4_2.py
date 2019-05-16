# -*- coding: utf-8 -*-
"""
Created on Thu May 16 02:15:59 2019

@author: 김진겸
"""

import tensorflow as tf
import numpy as np
xy=np.loadtxt("data.csv",delimiter="," , dtype=np.float32)
x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.matmul(X, W) + b
cost= tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    cost_val, hy_val,_=sess.run([cost,hypothesis, train],feed_dict={X:x_data,Y:y_data})
    if i%10==0:
        print(i,"Cost: ",cost_val, "\nPrediction:\n",hy_val)
print(sess.run(hypothesis,feed_dict={X:[[100,70,101]]}))

 