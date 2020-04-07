# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/12/30 16:52
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf
import json
from utils.radam import RAdamOptimizer


class CDSSM(object):

    def __init__(self,
                 vocab_file,
                 num_lstm_units,
                 vocab_size):
        self.vocab_file = vocab_file
        self.char_index = {' ': 0}
        self.load_dict()
        self.calss_size = 2
        self.unknow_char_id = len(self.char_index)
        self.vocab_size = len(self.char_index) + 1
        self.learning_rate = 0.0001
        self.num_lstm_units = num_lstm_units
        self.embedding_size = 256
        self.num_filters = 128
        self.vocab_size = vocab_size
        self.grad_clip = 5
        self.sequence = 70
        self.kernel_size = [2, 3, 4]
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        with tf.name_scope("dssm_lsm_declare"):
            self.queries = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence])
            self.docs = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence])
            self.y_data = tf.placeholder(tf.int32, shape=None, name='y_data')
            self.sequence = tf.placeholder(dtype=tf.int32, name='sequence')

        self.create_embedding()
        self.create_model()
        self.create_loss()

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def create_embedding(self):
        with tf.name_scope("sim_declare"):
            self.embed = tf.get_variable("embed", shape=[self.vocab_size, self.embedding_size],
                                         initializer=tf.random_normal_initializer(stddev=0.1))

    @staticmethod
    def cosine(p, h):
        p_norm = tf.norm(p, axis=1, keepdims=True)
        h_norm = tf.norm(p, axis=1, keepdims=True)
        cosine = tf.reduce_sum(tf.multiply(p, h), axis=1, keepdims=True) / (p_norm * h_norm)
        return cosine

    def create_model(self):
        self.embed_queries = tf.nn.embedding_lookup(self.embed, self.queries)
        self.embedding_expand_queries = tf.expand_dims(self.embed_queries, -1)
        self.embed_docs = tf.nn.embedding_lookup(self.embed, self.docs)
        self.embedding_expand_docs = tf.expand_dims(self.embed_docs, -1)
        with tf.variable_scope('query_cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.kernel_size):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # [kernel_size, 100, 1, 128]
                    # kernel_size个厚度为100的1*128的三维张量
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    # filter： w，要求也是一个张量，shape为[filter_height, filter_weight, in_channel, out_channels]，
                    # 其中 filter_height为卷积核高度，
                    # filter_weight为卷积核宽度
                    # in_channel是图像通道数 ，和input的in_channel要保持一致
                    # out_channel是卷积核数量。
                    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='w')
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')
                    # strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动一步，y方向运动一步
                    # padding采用的方式是VALID 输出高度 输入维度-kernel_size+1后除以步长
                    conv = tf.nn.conv2d(self.embedding_expand_queries, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                    pooled = tf.nn.max_pool(h, ksize=[1, self.sequence - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID', name='pool')
                    pooled_outputs.append(pooled)

            num_filter_total = self.num_filters * len(self.kernel_size)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat_1 = tf.reshape(self.h_pool, [-1, num_filter_total])
        with tf.variable_scope('doc_cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.kernel_size):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # [kernel_size, 100, 1, 128]
                    # kernel_size个厚度为100的1*128的三维张量
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    # filter： w，要求也是一个张量，shape为[filter_height, filter_weight, in_channel, out_channels]，
                    # 其中 filter_height为卷积核高度，
                    # filter_weight为卷积核宽度
                    # in_channel是图像通道数 ，和input的in_channel要保持一致
                    # out_channel是卷积核数量。
                    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='w')
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')
                    # strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动一步，y方向运动一步
                    # padding采用的方式是VALID 输出高度 输入维度-kernel_size+1后除以步长
                    conv = tf.nn.conv2d(self.embedding_expand_docs, w, strides=[1, 1, 1, 1], padding='VALID',
                                        name='conv')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                    pooled = tf.nn.max_pool(h, ksize=[1, self.sequence - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID', name='pool')
                    pooled_outputs.append(pooled)

            num_filter_total = self.num_filters * len(self.kernel_size)
            self.h_pool = tf.concat(pooled_outputs, 3)

            x = tf.layers.dense(self.h_pool, 256, activation='tanh')
            # x = self.dropout(x)
            # x = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2]))
            self.h_pool_flat_2 = tf.reshape(x, [-1, num_filter_total])

        outputs1 = tf.reduce_mean(self.h_pool_flat_1, axis=1)
        outputs2 = tf.reduce_mean(self.h_pool_flat_2, axis=1)

        pos_result = self.cosine(outputs1, outputs2)
        neg_result = 1 - pos_result
        self.logits = tf.concat([pos_result, neg_result], axis=1)
        self.prob = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.logits, axis=1)

    def create_loss(self):
        with tf.name_scope("siamese_loss"):
            y = tf.one_hot(self.y_data, self.calss_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
            self.loss = tf.reduce_mean(loss)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)

            self.train_op = RAdamOptimizer(self.learning_rate).minimize(self.loss)
            correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.y_data)
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.y_data, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32'), name='accuracy')