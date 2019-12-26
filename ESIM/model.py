# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/12/25 11:15
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf
import logging
from utils.radam import RAdamOptimizer

_logger = logging.getLogger()


class EsimCore:

    def __init__(self, io_sequence_size, vocab_size, class_size=2, learning_rate=0.001, trainable=False):

        # 为True表示训练
        self.is_training = trainable
        # 字个数
        self.vocab_size = vocab_size
        # 句子长度
        self.io_sequence_size = io_sequence_size
        # 学习率
        self.learning_rate = learning_rate
        # embedding维度
        self.embedding_size = 64
        # lstm单元数
        self.hidden_size = 128
        # 类别个数
        self.calss_size = class_size
        self.grad_clip = 5.0
        # dropout
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        with tf.variable_scope('esim_declare', reuse=tf.AUTO_REUSE):
            self.p = tf.placeholder(dtype=tf.int32, shape=(None, self.io_sequence_size), name='p')
            self.h = tf.placeholder(dtype=tf.int32, shape=(None, self.io_sequence_size), name='h')
            self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')
        self.create_embedding()
        self.create_model()
        if self.is_training is True:
            self.create_loss()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def create_embedding(self):
        with tf.variable_scope("esim_declare", reuse=tf.AUTO_REUSE):
            self.esim_variable = tf.get_variable("esim_embedding_variable",
                                                 shape=[self.vocab_size, self.embedding_size],
                                                 initializer=tf.random_normal_initializer(stddev=0.1))

    def create_model(self):
        # 第一个句子输入lstm
        with tf.variable_scope("esim_p", reuse=tf.AUTO_REUSE):
            self.embedded_layer = tf.nn.embedding_lookup(self.esim_variable, self.p)
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            (p_f, p_b), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.embedded_layer, dtype=tf.float32)
        # 第二个句子输入lstm
        with tf.variable_scope("esim_h"):
            self.embedded_layer = tf.nn.embedding_lookup(self.esim_variable, self.h)
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            (h_f, h_b), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.embedded_layer, dtype=tf.float32)

        # 合并前向后向的输出
        p = tf.concat([p_f, p_b], axis=2)
        h = tf.concat([h_f, h_b], axis=2)

        # 增加dropout
        p = self.dropout(p)
        h = self.dropout(h)

        # tf.transpose 转置，p与h转置相乘计算注意力
        # tf.matmul是矩阵乘法、tf.multiply是点乘
        e = tf.matmul(p, tf.transpose(h, perm=[0, 2, 1]))
        a_attention = tf.nn.softmax(e)
        b_attention = tf.transpose(tf.nn.softmax(tf.transpose(e, perm=[0, 2, 1])), perm=[0, 2, 1])

        # 基于注意力利用h表示p、利用p表示h
        a = tf.matmul(a_attention, h)
        b = tf.matmul(b_attention, p)

        # 计算ap的点乘，ap的差等组合成新的向量
        m_a = tf.concat((a, p, a - p, tf.multiply(a, p)), axis=2)
        m_b = tf.concat((b, h, b - h, tf.multiply(b, h)), axis=2)

        # 新的向量传入lstm进行特征提取
        with tf.variable_scope("esim_a", reuse=tf.AUTO_REUSE):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            (a_f, a_b), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, m_a, dtype=tf.float32)
        with tf.variable_scope("esim_b", reuse=tf.AUTO_REUSE):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            (b_f, b_b), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, m_b, dtype=tf.float32)

        a = tf.concat((a_f, a_b), axis=2)
        b = tf.concat((b_f, b_b), axis=2)

        a = self.dropout(a)
        b = self.dropout(b)

        # 计算平均池化和最大池化
        a_avg = tf.reduce_mean(a, axis=2)
        b_avg = tf.reduce_mean(b, axis=2)

        a_max = tf.reduce_max(a, axis=2)
        b_max = tf.reduce_max(b, axis=2)

        v = tf.concat((a_avg, a_max, b_avg, b_max), axis=1)
        # 全连接层
        v = tf.layers.dense(v, 512, activation=tf.nn.tanh)
        v = self.dropout(v)
        self.logits = tf.layers.dense(v, 2, activation=tf.nn.tanh)
        self.prob = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.logits, axis=1)

    def create_loss(self):
        with tf.variable_scope("esim_loss", reuse=tf.AUTO_REUSE):

            y = tf.one_hot(self.y, self.calss_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
            self.loss = tf.reduce_mean(loss)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)

            # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.train_op = RAdamOptimizer(self.learning_rate).minimize(self.loss)
            correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.y)
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))