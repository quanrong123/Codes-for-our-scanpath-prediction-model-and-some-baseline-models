import tensorflow as tf

def read_data():
    print("read data ...")
    return tf.constant(value=[1.0, 2.0, 3.0], dtype=tf.float32)

X = read_data()
X_train = tf.placeholder(dtype=tf.float32)
with tf.Session() as sess:
    for epoch in range(3):
        for batch in range(3):
            x = sess.run(X)
            print(sess.run(X_train, feed_dict={X_train: x}))
            # suiran xunhuantinei sess.run(X) le 9 ci, dan read_data shiji quezhidiaoyongleyici

def read_data():
    print("read data ...")
    return tf.random_uniform(shape=(3,), maxval=1)

X = read_data()
X_train = tf.placeholder(dtype=tf.float32)
with tf.Session() as sess:
    for epoch in range(3):
        for batch in range(3):
            x = sess.run(X)
            print(sess.run(X_train, feed_dict={X_train: x}))

def hello():
    print("hello")

def read_data():
    print("read data ...")
    print("load 10G data")
    print("[1,2,3,4,...,99999]")
    hello()
    return tf.random_uniform(shape=(3,), maxval=1.0)

X = read_data()
X_train = tf.placeholder(dtype=tf.float32)












