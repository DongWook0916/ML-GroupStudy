import tensorflow as tf
tf.set_random_seed(777)  

x_data = [[1, 2],[2, 3],[3, 1],[4, 3],[5, 3],[6, 2]]
y_data = [[0],[0],[0],[1],[1],[1]]

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])
#n개일 수 있으니 none

w = tf.Variable(tf.random_normal([2, 1]), name='weight')
#매트릭스 X*W [input,output] bias는 항상 output 개
b = tf.Variable(tf.random_normal([1]), name='bias')

# H(X) 대입
h = tf.sigmoid(tf.matmul(x,w) + b)

# cost(x)
c = -tf.reduce_mean(y * tf.log(h) + (1 - y) *tf.log(1 - h))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(c)
#최소 찾기!

#예측한 값을 0이나 1로 처리(cast로 true false로 반환), 정확도는 평균으로 파악
predicted = tf.cast(h > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# 학습하기
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_val, _ = sess.run([c, train], feed_dict={x: x_data, y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # 정확도 검사
    hy, c, a = sess.run([h, predicted, accuracy],
                       feed_dict={x: x_data, y: y_data})
    print("\nHypothesis: ", hy, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
    #정확하다....

