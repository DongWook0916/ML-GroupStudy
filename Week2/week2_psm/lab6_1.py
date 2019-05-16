import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1, 2, 1, 1],[2, 1, 3, 2],[3, 1, 3, 4],[4, 1, 5, 5],[1, 7, 5, 5],[1, 2, 5, 6],[1, 6, 6, 6],[1, 7, 7, 7]]
y_data = [[0, 0, 1],[0, 0, 1],[0, 0, 1],[0, 1, 0],[0, 1, 0],[0, 1, 0],[1, 0, 0],[1, 0, 0]]

x = tf.placeholder("float", [None, 4])
y = tf.placeholder("float", [None, 3])
nb_classes = 3

w = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

h = tf.nn.softmax(tf.matmul(x, w) + b)

c = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(c)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
            _, cost_val = sess.run([optimizer, c], feed_dict={x: x_data, y: y_data})

            if step % 200 == 0:
                print(step, cost_val)

    print('--------------')
    a = sess.run(h, feed_dict={x: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    print('--------------')
    b = sess.run(h, feed_dict={x: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print('--------------')
    c = sess.run(h, feed_dict={x: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))

    print('--------------')
    all = sess.run(h, feed_dict={x: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))

