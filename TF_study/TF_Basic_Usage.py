#https://www.tensorflow.org/get_started/basic_usage

# Represents computations as graphs. GRAPH
# Executes graphs in the context of Sessions. SESSION
# Represents data as tensors. TENSOR
# Maintains state with Variables. VARIABLE
# Uses feeds and fetches to get data into and out of arbitrary operations.　FEED FETCH

import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

with tf.Session() as sess:
    result = sess.run([product])
    print(result)

# with tf.Session() as sess :
#/cpu:0
#/gpu:0
#/gpu:1
#     with tf.device("/gpu:1") :
#         matrix1 = tf.constant([[3., 3.]])
#         matrix2 = tf.constant([[2.], [2.]])
#         product = tf.matmul(matrix1, matrix2)
#         ...

#with tf.Session("grpc://example.org:2222") as sess:
  # Calls to sess.run(...) will be executed on the cluster.

# with tf.device("/job:ps/task:0"):
#   weights = tf.Variable(...)
#   biases = tf.Variable(...)

# Enter an interactive TensorFlow Session.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()

# Add an op to subtract 'a' from 'x'.  Run it and print the result
sub = tf.sub(x, a)
print(sub.eval())
# ==> [-2. -1.]

# Close the Session when we're done.
sess.close()

state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3) :
        sess.run(update)
        print(sess.run(state))

input1 = tf.constant([3.])
input2 = tf.constant([2.])
input3 = tf.constant([5.])
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess :
    result = sess.run([mul, intermed])
    print(result)

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))