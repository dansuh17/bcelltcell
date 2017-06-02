import sys
from tomography import SampleGenerator
import tensorflow as tf
import numpy as np


# constants
__EPOCHS__ = 200
__INIT_LEARNING_RATE__ = 0.003
__DECAY_STEPS__ = 700  # decays once per this amount of iterations
__DECAY_RATE__ = 0.93
__BATCH_SIZE__ = 30
__KEEP_PROB_CONV__ = 0.8
__KEEP_PROB_FC__ = 0.7
__NUM_MODELS__ = 5


class Model:
    """Defines a single model."""
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.construct_net()

    def train(self, batch_x, batch_y):
        # TODO: merged should be run separately...i guess
        return self.sess.run([self.optimizer],
                             feed_dict={self.x: batch_x, self.y: batch_y})

    def glob_step(self, batch_x, batch_y):
        return self.sess.run([self.global_step],
                feed_dict={self.x: batch_x, self.y: batch_y})

    def calc_accuracy(self, batch_x, batch_y):
        return self.sess.run([self.accuracy, self.corrects], feed_dict={self.x: batch_x, self.y: batch_y})

    def calc_loss(self, batch_x, batch_y):
        return self.sess.run([self.loss], feed_dict={self.x: batch_x, self.y: batch_y})

    def predict(self, batch_x, batch_y):
        return self.sess.run([self.logits], feed_dict={self.x: batch_x, self.y: batch_y})

    def construct_net(self):
        with tf.variable_scope(self.name):  # create a scope by its name
            # (batch, in_height, in_width, in_channels)
            self.x = tf.placeholder(tf.float32, [None, 66, 66, 9], name='x')
            self.y = tf.placeholder(tf.int64, [None], name='y')
            images = tf.placeholder(tf.float32, shape=[1, 17, 17])  # for viz


            with tf.variable_scope('conv1') as scope:
                out_filters = 64
                kernel = tf.get_variable('wieghts', [5, 5, 9, out_filters], tf.float32,
                                        tf.truncated_normal_initializer(stddev=0.1))
                conv = tf.nn.conv2d(self.x, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
                mean, var = tf.nn.moments(conv, [0, 1, 2])
                conv = tf.nn.batch_normalization(conv, mean, var, 0, 1, 0.0001)  # BN
                biases = tf.get_variable('biases', [out_filters], tf.float32,
                                        tf.constant_initializer(0.1, dtype=tf.float32))
                bias_added = tf.nn.bias_add(conv, biases)
                # activate
                conv1 = tf.nn.relu(bias_added, name=scope.name)
                tf.summary.histogram("activations" + self.name, conv1)

            # max-pooling
            # TODO: ksize? strides?
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME')
            pool1_dropout = tf.nn.dropout(pool1, keep_prob=__KEEP_PROB_CONV__)
            in_filters = out_filters
            print('conv1 layer ready')

            with tf.variable_scope('conv2') as scope:
                out_filters = 32
                kernel = tf.get_variable('weights', [3, 3, in_filters, out_filters],
                                        tf.float32, tf.truncated_normal_initializer(stddev=0.1))
                conv = tf.nn.conv2d(pool1_dropout, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
                mean, var = tf.nn.moments(conv, [0, 1, 2])
                conv = tf.nn.batch_normalization(conv, mean, var, 0, 1, 0.0001)  # BN
                biases = tf.get_variable('biases', [out_filters], tf.float32,
                                        tf.constant_initializer(0.1, dtype=tf.float32))
                bias_added = tf.nn.bias_add(conv, biases)
                conv2 = tf.nn.relu(bias_added, name=scope.name) # activate
                tf.summary.histogram("activations" + self.name, conv2)
            print('conv2 layer ready')

            # max pooling
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool2_dropout = tf.nn.dropout(pool2, keep_prob=__KEEP_PROB_CONV__)
            in_filters = out_filters

            with tf.variable_scope('conv3') as scope:
                out_filters = 16
                kernel = tf.get_variable('weights', [3, 3, in_filters, out_filters],
                                        tf.float32, tf.truncated_normal_initializer(stddev=0.1))
                conv = tf.nn.conv2d(pool2_dropout, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
                mean, var = tf.nn.moments(conv, [0, 1, 2])
                conv = tf.nn.batch_normalization(conv, mean, var, 0, 1, variance_epsilon=0.0001)  # BN
                biases = tf.get_variable('biases', [out_filters], tf.float32,
                                        tf.constant_initializer(0.1, dtype=tf.float32))
                bias_added = tf.nn.bias_add(conv, biases)
                conv3 = tf.nn.relu(bias_added, name=scope.name)  # activate
                tf.summary.histogram("activations" + self.name, conv3)
                conv3_dropout = tf.nn.dropout(conv3, keep_prob=__KEEP_PROB_CONV__)
            print('conv3 layer ready')

            # save sample image - let's try...
            conv3_out_shape = conv3_dropout.get_shape().as_list() # ?, 17, 17, 16
            conv3_out_shape = conv3_out_shape[1:-1]
            conv3_out_shape.append(1)
            conv3_out_shape.insert(0, 1)
            slice_image = tf.slice(conv3_dropout, [1, 0, 0, 0], conv3_out_shape)
            tf.summary.image('nn_out_image' + self.name, slice_image, max_outputs=1)


            # fully connected layers
            with tf.variable_scope('fc1') as scope:
                fc_size = 1024
                # flatten dimensions except batch size
                dim = np.prod(conv3_dropout.get_shape().as_list()[1:])
                prev_layer_flat = tf.reshape(conv3_dropout, [-1, dim])  # vectorize

                weights = tf.get_variable('weights', [dim, fc_size])
                biases = tf.get_variable('biases', [fc_size])
                fc1_out = tf.nn.relu(tf.add(tf.matmul(prev_layer_flat, weights), biases))
                tf.summary.histogram("fc_out" + self.name, fc1_out)
                fc1_out_dr = tf.nn.dropout(fc1_out, keep_prob=__KEEP_PROB_FC__)
            print('fully connected layer 1 ready')
            fc_in = fc_size

            with tf.variable_scope('fc2') as scope:
                fc_out = 512
                weights = tf.get_variable('weights', [fc_in, fc_out])
                biases = tf.get_variable('biases', [fc_out])
                fc2_out = tf.nn.relu(tf.add(tf.matmul(fc1_out_dr, weights), biases))
                tf.summary.histogram("fc_out" + self.name, fc2_out)
                fc2_out_dr = tf.nn.dropout(fc2_out, keep_prob=__KEEP_PROB_FC__)
            print('fully connected layer 2 ready')
            fc_in = fc_out

            with tf.variable_scope('fc3') as scope:
                num_classes = 3
                weights = tf.get_variable('weights', [fc_in, num_classes])
                biases = tf.get_variable('biases', [num_classes])
                fc3_out = tf.add(tf.matmul(fc2_out_dr, weights), biases, name='logits')  # no relu, no dropout
                tf.summary.histogram("fc_out" + self.name, fc3_out)
            print('fully connected layer 3 ready')

            # save logits as its member variable
            self.logits = fc3_out

            # loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=fc3_out, labels=self.y)
            self.loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss' + self.name, self.loss)  # save summary of loss

            # optimize
            # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            self.global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = __INIT_LEARNING_RATE__
            self.learning_rate = tf.train.exponential_decay(
                    starter_learning_rate, self.global_step,
                    decay_steps=__DECAY_STEPS__,
                    decay_rate=__DECAY_RATE__, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate) \
                .minimize(self.loss, global_step=self.global_step)
            tf.summary.scalar('learning_rate' + self.name, self.learning_rate)
            print('Optimizer Ready')

            # evaluate
            self.corrects = tf.equal(tf.argmax(fc3_out, axis=1), self.y, name='corrects')
            self.accuracy = tf.reduce_mean(tf.cast(self.corrects, tf.float32), name='accuracy')
            tf.summary.scalar('accuracy' + self.name, self.accuracy)  # save summary of accuracy
            print('Network ready!')

if __name__ == '__main__':
    # inference testing
    if len(sys.argv) > 1 and sys.argv[1] == 'infer':
        with tf.Session() as sess:
            # restore trained graph
            saver = tf.train.import_meta_graph('./model/model_2dcnn_ensemble.ckpt.meta')
            saver.restore(sess, './model/model_2dcnn_ensemble.ckpt')

            graph = tf.get_default_graph()
            # print([op.name for op in graph.get_operations()])
            # print([var.name for var in tf.global_variables()])

            # collect ops
            correct_ops = []
            acc_ops = []
            xes = []
            ys = []
            for i in range(__NUM_MODELS__):
                correct_op = graph.get_operation_by_name('model_{}/corrects'.format(i))
                acc_op = graph.get_operation_by_name('model_{}/accuracy'.format(i))
                # example result of .outputs[0] : Tensor("model_0/Placeholder:0", shape=(?, 66, 66, 9), dtype=float32)
                x_placeholder = graph.get_operation_by_name('model_{}/x'.format(i)).outputs[0]
                y_placeholder = graph.get_operation_by_name('model_{}/y'.format(i)).outputs[0]

                correct_ops.append(correct_op)
                acc_ops.append(acc_op)
                xes.append(x_placeholder)
                ys.append(y_placeholder)

            # ensemble testing
            print('Testing ensemble')
            sg = SampleGenerator(filename='augmented_dataset_nowater.h5',
                                batch_size=__BATCH_SIZE__,
                                use_original_sets=False)
            # generate test data
            test_x, test_y = sg.test_sample_slices(random_sample=None)
            test_size = len(test_y)
            print('test size : {}'.format(test_size))
            sum_predictions = np.zeros(test_size)

            # create feed dict
            feed_dict_ = {}
            for m_idx in range(__NUM_MODELS__):
                feed_dict_[xes[m_idx]] = test_x
                feed_dict_[ys[m_idx]] = test_y

            # sum the predited values from all models
            for m_idx in range(__NUM_MODELS__):
                corrects = sess.run(acc_ops[m_idx].outputs[0], feed_dict=feed_dict_)
                predictions = sess.run(correct_ops[m_idx].outputs[0], feed_dict=feed_dict_)
                print(np.array(predictions).astype(int))
                print('Accuracy for model {} : {} / {}'.format(m_idx, np.sum(predictions), test_size))
                sum_predictions += predictions

            # retrieve the argmax of sum of predictions from all models
            # ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), test_y) # the argmax val should be equal to the label
            # ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))

            # simply do a majority vote
            ensemble_accuracy = sum_predictions > (__NUM_MODELS__ / 2)
            total_corrects = np.sum(ensemble_accuracy)
            print('Ensemble accuracy: {} / {}'.format(total_corrects, test_size))
    else:
        # simply train when no arguments are given
        # create a session
        with tf.Session() as sess:
            # create models
            models = []
            for model_idx in range(__NUM_MODELS__):
                models.append(Model(sess, 'model_{}'.format(model_idx)))
            print('Models Ready')

            # create a saver to save the model
            saver = tf.train.Saver()
            print('Saver initialized')

            # merge summaries and initialize writers
            merged = tf.summary.merge_all()  # merge summary - does this do for all models?
            train_writer = tf.summary.FileWriter(
                    './summaries/train_dropout_nowater_ensemble', sess.graph)
            test_writer = tf.summary.FileWriter('./summaries/test_dropout_nowater_ensemble')
            print('Summary writers ready')

            # 'augmented_dataset_2.h5'
            sg = SampleGenerator(filename='augmented_dataset_nowater.h5',
                                batch_size=__BATCH_SIZE__,
                                use_original_sets=False)  # force using original sets
            print('Samples ready')

            # training
            print('Begin session')
            init = tf.global_variables_initializer()
            sess.run(init)

            # restoring model:
            # saver.restore(sess, './model/model.ckpt')
            # print('Model restored')
            for epoch in range(__EPOCHS__):
                # refresh samples as new epoch begins
                sg.reset_index()
                print('epoch : {}'.format(epoch))

                # for batch iterations
                for batch_iter in range(sg.num_batches):
                    batch_x, batch_y = sg.generate_sample_slices()

                    for model in models:
                        _ = model.train(batch_x, batch_y)

                    # global step
                    g_step = models[0].glob_step(batch_x, batch_y)[0]

                    # for every mod5 == 0, print out the loss / accuracy
                    if g_step % 5 == 0:
                        losses = []
                        test_losses = []
                        train_accs = []
                        train_corrects = []
                        test_accs = []
                        test_corrects = []

                        # generate test samples
                        test_x, test_y = sg.test_sample_slices()

                        for model in models:
                            # train loss / accuracy
                            loss_val = model.calc_loss(batch_x, batch_y)
                            losses.append(loss_val)  # loss values for printing
                            train_acc, train_correct = model.calc_accuracy(batch_x, batch_y)
                            train_accs.append(train_acc)
                            train_corrects.append(train_correct)

                            # test loss / acc
                            test_loss = model.calc_loss(test_x, test_y)
                            test_acc, test_correct = model.calc_accuracy(test_x, test_y)
                            test_losses.append(test_loss)
                            test_accs.append(test_acc)
                            test_corrects.append(test_correct)

                        # print the results
                        print('::: For step {} , epoch {} ::: '.format(g_step, epoch))
                        print('train loss')
                        print(losses)
                        print('train accuracies')
                        print(train_accs)
                        for idx in range(len(models)):
                            print('[{} / {}] '.format(
                                np.sum(train_corrects[idx]), len(train_corrects[idx])), end='')
                        print('')

                        print('test loss')
                        print(test_losses)
                        print('test accuracies')
                        print(test_accs)
                        print('test corrects')
                        for idx in range(len(models)):
                            print('[{} / {}] '.format(
                                np.sum(test_corrects[idx]), len(test_corrects[idx])), end='')
                        print('\n')

                        # add test summaries
                        feed_dict_test = {}
                        for m in models:
                            feed_dict_test[m.x] = test_x
                            feed_dict_test[m.y] = test_y
                        summ_test = sess.run(merged, feed_dict=feed_dict_test)
                        test_writer.add_summary(summ_test, g_step)

                    # store train summaries per iteration
                    feed_dict_ = {}
                    for m in models:
                        feed_dict_[m.x] = batch_x
                        feed_dict_[m.y] = batch_y
                    summ = sess.run(merged, feed_dict=feed_dict_)
                    train_writer.add_summary(summ, g_step)  # write summary
            print('Training complete')

            # save the trained model
            save_path = saver.save(sess, './model/model_2dcnn_ensemble.ckpt')
            print('Model saved at : {}'.format(save_path))
