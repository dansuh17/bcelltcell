import tensorflow as tf
import numpy as np
from tomography import SampleGenerator

# PREPARE THE NETWORK

# (batch, in_depth, in_height, in_width, in_channels]
x = tf.placeholder(tf.float32, [None, 66, 66, 66, 1])
y = tf.placeholder(tf.int64, [None])  # classification btwn. three

FC_SIZE = 1024
DTYPE = tf.float32


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE,
                           tf.truncated_normal_initializer(stddev=0.1))


def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE,
                           tf.constant_initializer(0.1, dtype=DTYPE))


with tf.variable_scope('conv1') as scope:
    out_filters = 16
    kernel = _weight_variable('weights', [5, 5, 5, 1, out_filters])
    conv = tf.nn.conv3d(x, filter=kernel,
                        strides=[1, 1, 1, 1, 1], padding='SAME')
    biases = _bias_variable('biases', [out_filters])
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)

    prev_layer = conv1
    in_filters = out_filters

pool1 = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1],
                         strides=[1, 2, 2, 2, 1], padding='SAME')
norm1 = pool1  # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')

prev_layer = norm1

print('conv1 layer ready')
with tf.variable_scope('conv2') as scope:
    out_filters = 32
    kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
    conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
    biases = _bias_variable('biases', [out_filters])
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)

    prev_layer = conv2
    in_filters = out_filters

# pool prev_layer here
# TODO: batch normalization?
prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1], padding='SAME')

print('conv2 layer ready')
with tf.variable_scope('conv3_1') as scope:
    out_filters = 64
    kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
    conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
    biases = _bias_variable('biases', [out_filters])
    bias = tf.nn.bias_add(conv, biases)
    prev_layer = tf.nn.relu(bias, name=scope.name)
    in_filters = out_filters

print('conv3-1 layer ready')
with tf.variable_scope('conv3_2') as scope:
    out_filters = 64
    kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
    conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
    biases = _bias_variable('biases', [out_filters])
    bias = tf.nn.bias_add(conv, biases)
    prev_layer = tf.nn.relu(bias, name=scope.name)
    in_filters = out_filters

print('conv3-2 layer ready')
with tf.variable_scope('conv3_3') as scope:
    out_filters = 32
    kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
    conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
    biases = _bias_variable('biases', [out_filters])
    bias = tf.nn.bias_add(conv, biases)
    prev_layer = tf.nn.relu(bias, name=scope.name)
    in_filters = out_filters

print('conv3-3 layer ready')
# normalize prev_layer here
prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1], padding='SAME')

with tf.variable_scope('local3') as scope:
    dim = np.prod(prev_layer.get_shape().as_list()[1:])
    prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
    weights = _weight_variable('weights', [dim, FC_SIZE])
    biases = _bias_variable('biases', [FC_SIZE])
    local3 = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)

prev_layer = local3

print('local3 layer ready')
with tf.variable_scope('local4') as scope:
    dim = np.prod(prev_layer.get_shape().as_list()[1:])
    prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
    weights = _weight_variable('weights', [dim, FC_SIZE])
    biases = _bias_variable('biases', [FC_SIZE])
    local4 = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)

prev_layer = local4
print('local4 layer ready')

with tf.variable_scope('softmax_linear') as scope:
    dim = np.prod(prev_layer.get_shape().as_list()[1:])
    weights = _weight_variable('weights', [dim, 3])
    biases = _bias_variable('biases', [3])  # num_classes
    softmax_linear = tf.add(tf.matmul(prev_layer, weights), biases, name=scope.name)

print('softmax ready')

# define loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=softmax_linear, labels=y)
loss = tf.reduce_mean(cross_entropy)
tf.summary.scalar('loss', loss)  # save loss

# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.3).minimize(loss)

# calculate accuracy for display
predicted = tf.argmax(softmax_linear, axis=1)
_corr = tf.equal(predicted, y)  # Count corrects
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))  # Accuracy
tf.summary.scalar('accruracy', accr)  # save accuracy

print('Network ready!')


if __name__ == '__main__':
    with tf.Session() as sess:
        saver = tf.train.Saver()
        print('Saver initialized')

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./summaries/train_3d', sess.graph)
        test_writer = tf.summary.FileWriter('./summaries/test_3d')
        print('Begin session')
        sess.run(tf.global_variables_initializer())

        print('Init complete')
        avg_cost = 0
        global_step = 0
        for epoch in range(200):
            # refresh samples as new epoch begins
            sg = SampleGenerator(filename='augmented_dataset_nowater.h5', batch_size=20)
            sg.reset_index()
            print('epoch : {}'.format(epoch))

            # for batch iterations
            for batch_iter in range(sg.num_batches):
                global_step += 1
                batch_x, batch_y = sg.generate_samples()
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                summary, loss_val = sess.run([merged, loss], feed_dict={x: batch_x, y: batch_y})
                train_writer.add_summary(summary, global_step)  # add train summary

                if batch_iter % 5 == 0:
                    # for training data
                    print('loss : {}'.format(loss_val))
                    train_acc = sess.run(accr, feed_dict={x: batch_x, y: batch_y})
                    print('train_acc : {}'.format(train_acc))

                    # display test data information
                    test_x, test_y = sg.test_samples()
                    test_summary, test_acc, test_correct = sess.run([merged, accr, _corr], feed_dict={x: test_x, y: test_y})
                    test_writer.add_summary(test_summary, global_step)  # add test summary
                    print('test_acc : {}'.format(test_acc))
                    print('test_correct : {} / {}'.format(np.sum(test_correct), len(test_correct)))
                    print('')

        # save the model once the training is done
        saved_path = saver.save(sess, './model/model_3dcnn.ckpt')
        print('Training Completed')
        print('Model saved at: {}'.format(saved_path))

        print('Testing for all test sets')
        test_x, test_y = sg.test_samples(random_sample=None)
        test_acc, test_corr, test_pred = sess.run(
                [accr, _corr, predicted], feed_dict={x: test_x, y: test_y})

        print('Test Accuracy')
        print(accr)
        print('Test Corrects')
        print(test_corr)
        print('Test Predictions')
        print(test_pred)

