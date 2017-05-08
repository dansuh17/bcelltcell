import tensorflow as tf
import h5py
import numpy as np


class SampleGenerator:
    def __init__(self, filename, batch_size):
        self.filename = filename
        self.batch_size = batch_size

        train_sample_ids, test_sample_ids = self.split_dataset(ratio=0.96)
        num_train_data = len(train_sample_ids)
        self.num_batches = num_train_data // self.batch_size

        # trim the leftovers
        train_sample_ids = train_sample_ids[:(self.num_batches * self.batch_size)]
        self.train_sample_sets = np.split(train_sample_ids, self.num_batches)
        self.test_sample_ids = test_sample_ids
        self.batch_index = 0

        print('Train samples : {}'.format(len(train_sample_ids)))
        print('Test samples : {}'.format(len(test_sample_ids)))

    def reset_index(self):
        self.batch_index = 0

    def split_dataset(self, ratio):
        with h5py.File(self.filename, 'r') as hf:
            data_nums = [num for num in hf]
        num_samples = len(data_nums)
        split_point = int(num_samples * ratio)
        return np.split(np.random.permutation(data_nums), [split_point])

    def batch_and_label(self, id_list):
        with h5py.File(self.filename, 'r') as hf:
            batch_data = []
            label_data = []

            for samp_id in id_list:
                sample = np.array(hf[samp_id]['data'])
                sample = sample.reshape(sample.shape + (1, ))
                batch_data.append(sample)

                label = hf[samp_id]['data'].attrs['label']
                # label_flag = [0, 0, 0]
                if label == 'B':
                    label_idx = 0
                elif label == 'CD4':
                    label_idx = 1
                elif label == 'CD8':
                    label_idx = 2
                else:
                    raise ValueError
                label_data.append(label_idx)
        return batch_data, label_data

    def generate_samples(self):
        set_ = self.train_sample_sets[self.batch_index]
        self.batch_index += 1
        return self.batch_and_label(set_)

    def test_samples(self):
        return self.batch_and_label(self.test_sample_ids)


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

# normalize prev_layer here
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

# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.5).minimize(loss)

# calculate accuracy for display
_corr = tf.equal(tf.argmax(softmax_linear, axis=1), y)  # Count corrects
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))  # Accuracy

print('Network ready!')

if __name__ == '__main__':
    # batch, label = sg.generate_samples()
    # print(batch, label)

    with tf.Session() as sess:
        print('Begin session')
        sess.run(tf.global_variables_initializer())

        print('Init complete')
        avg_cost = 0
        for epoch in range(20):
            # refresh samples as new epoch begins
            sg = SampleGenerator(filename='augmented_dataset.h5', batch_size=10)
            sg.reset_index()
            print('epoch : {}'.format(epoch))

            # for batch iterations
            for batch_iter in range(sg.num_batches):
                batch_x, batch_y = sg.generate_samples()
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                loss_val = sess.run(loss, feed_dict={x: batch_x, y: batch_y})

                if batch_iter % 5 == 0:
                    print('loss : {}'.format(loss_val))
                    train_acc = sess.run(accr, feed_dict={x: batch_x, y: batch_y})
                    print('train_acc : {}'.format(train_acc))
                    test_x, test_y = sg.test_samples()
                    test_acc, test_correct = sess.run([accr, _corr], feed_dict={x: test_x, y: test_y})
                    print('test_acc : {}'.format(test_acc))
                    print('test_correct : {} / {}'.format(np.sum(test_correct), len(test_correct)))
                    print('')
