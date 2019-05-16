# -*- coding: utf-8 -*-
"""
Created on Thu May 16 01:24:44 2019

@author: 김진겸
"""
import tensorflow as tf
#tf.set_ranomd_seed(777)

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

x1=tf.placeholder(tf.float32)
x2=tf.placeholder(tf.float32)
x3=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)


w1=tf.Variable(tf.random_normal([1]),name="weight1")
w2=tf.Variable(tf.random_normal([1]),name="weight2")
w3=tf.Variable(tf.random_normal([1]),name="weight3")
b=tf.Variable(tf.random_normal([1]),name="bias")

hypo=x1*w1+x2*w2+x3*w3+b
cost=tf.reduce_mean(tf.square(hypo-y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()

sess.run(tf.global_variables_initializer())
for i in range(2001):
    cost_val, hy_val,_=sess.run([cost,hypo, train],feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,y:y_data})
    if i%10==0:
        print(i,"Cost: ",cost_val, "\nPrediction:\n",hy_val)
        
"""Using matrix"""
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

#정규 분포 난수 생성 (3,1)-3*1shape으
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
B = tf.Variable(tf.random_normal([1]), name='biass')

# mtmul(x,y)-x와 y의 행렬 곱
hypothesis = tf.matmul(X, W) + B
co = tf.reduce_mean(tf.square(hypothesis - Y))
#경험에 의해 보통 1e-5사용하는거...
optim = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
tr = optim.minimize(co)
s = tf.Session()
s.run(tf.global_variables_initializer())


for step in range(2001):
    cost_val, hy_val, _ = s.run(
        [co, hypothesis, tr], feed_dict={X:x_data, Y:y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)










