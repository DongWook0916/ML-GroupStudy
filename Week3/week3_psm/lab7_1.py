import tensorflow as tf
tf.set_random_seed(777)
x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_test=[[2,1,1],[3,1,2],[3,3,4]]
y_test=[[0,0,1],[0,0,1],[0,0,1]]

x=tf.placeholder("float",[None,3])
y=tf.placeholder("float",[None,3])
w=tf.Variable(tf.random_normal([3,3]))
b=tf.Variable(tf.random_normal([3]))

h = tf.nn.softmax(tf.matmul(x ,w) + b)
c = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), axis=1))
op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(c)

p = tf.argmax(h, 1)
is_correct = tf.equal(p, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, w_val, _ = sess.run([c, w, op], feed_dict={x: x_data, y: y_data})
        print(step, cost_val, w_val)

    print("Prediction:", sess.run(p, feed_dict={x: x_test}))
    print("Accuracy: ", sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
