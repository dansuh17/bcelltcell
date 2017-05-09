from tomography import SampleGenerator
import tensorflow as tf
import numpy as np

# (batch, in_height, in_width, in_channels)
x = tf.placeholder(tf.float32, [None, 66, 66, 9])
y = tf.placeholder(tf.int64, [None])


with tf.variable_scope('conv1') as scope:
    out_filters = 64
    kernel = tf.get_variable('wieghts', [5, 5, 9, out_filters], tf.float32,
                             tf.truncated_normal_initializer(stddev=0.1))
    conv = tf.nn.conv2d(x, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [out_filters], tf.float32,
                             tf.constant_initializer(0.1, dtype=tf.float32))
    bias_added = tf.nn.bias_add(conv, biases)
    # activate
    conv1 = tf.nn.relu(bias_added, name=scope.name)  # TODO: name??

# max-pooling
# TODO: ksize? strides?
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')
in_filters = out_filters
print('conv1 layer ready')

with tf.variable_scope('conv2') as scope:
    out_filters = 16
    kernel = tf.get_variable('weights', [3, 3, in_filters, out_filters],
                             tf.float32, tf.truncated_normal_initializer(stddev=0.1))
    conv = tf.nn.conv2d(pool1, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [out_filters], tf.float32,
                             tf.constant_initializer(0.1, dtype=tf.float32))
    bias_added = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias_added, name=scope.name)
print('conv2 layer ready')

# fully connected
with tf.variable_scope('fc1') as scope:
    fc_size = 3  # num-classes!
    dim = np.prod(conv2.get_shape().as_list()[1:])
    prev_layer_flat = tf.reshape(conv2, [-1, dim])
    weights = tf.get_variable('weights', [dim, fc_size])
    biases = tf.get_variable('biases', [fc_size])
    out = tf.add(tf.matmul(prev_layer_flat, weights), biases)
print('fully connected layer ready')

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=out, labels=y)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

corrects = tf.equal(tf.argmax(out, axis=1), y)
accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
print('Network ready!')

with tf.Session() as sess:
    print('Begin session')
    init = tf.global_variables_initializer()
    sess.run(init)

    print('Init complete')
    for epoch in range(20):
        # refresh samples as new epoch begins
        sg = SampleGenerator(filename='augmented_dataset.h5', batch_size=10)
        sg.reset_index()
        print('epoch : {}'.format(epoch))

        # for batch iterations
        for batch_iter in range(sg.num_batches):
            batch_x, batch_y = sg.generate_sample_slices()
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            if batch_iter % 5 == 0:
                loss_val = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
                print('loss : {}'.format(loss_val))
                test_x, test_y = sg.test_sample_slices()
                test_acc, test_correct = sess.run([accuracy, corrects],
                                                  feed_dict={x: test_x, y: test_y})
                print('test_acc : {}'.format(test_acc))
                print('test_correct : {} / {}'.format(np.sum(test_correct),
                                                      len(test_correct)))
                print('')
