# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/12/20 18:01
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf
import logging
from utils.radam import RAdamOptimizer

_logger = logging.getLogger()


class DssmCore:
    """相当于伪孪生网络，两个网络之间不共享参数，各自学习"""

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
        self.embedding_size = 256
        # lstm单元数
        self.hidden_size = 512
        # 类别个数
        self.calss_size = class_size
        self.grad_clip = 5.0
        # dropout
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        with tf.variable_scope('dssm_declare', reuse=tf.AUTO_REUSE):
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
        # tf.AUTO_REUSE：如果创建过就返回，没有创建过就创建一个新的变量返回
        with tf.variable_scope("dssm_declare", reuse=tf.AUTO_REUSE):
            self.embedding = tf.get_variable(dtype=tf.float32, shape=(self.vocab_size, self.embedding_size),
                                             name='dssm_embedding', trainable=True)

    def cosine(self, p, h):
        # 用于计算向量，矩阵和tensor的范数，默认情况下是计算欧氏距离的L2范数
        # keep_dims是否保持维度不变
        p_norm = tf.norm(p, axis=1, keepdims=True)
        h_norm = tf.norm(p, axis=1, keepdims=True)
        # 公式：http://note.youdao.com/noteshare?id=b04ad8ab7211331073217e8202056585&sub=44E9A792C5D04903A73118503699E7D9
        cosine = tf.reduce_sum(tf.multiply(p, h), axis=1, keepdims=True) / (p_norm * h_norm)
        # 将结果限制在固定范围
        cos_sim_prob = tf.clip_by_value(cosine, 1e-8, 1.0)

        # cosine = 0.5 + 0.5 * cosine

        return cos_sim_prob

    def create_model(self):
        p_embedding = tf.nn.embedding_lookup(self.embedding, self.p)
        h_embedding = tf.nn.embedding_lookup(self.embedding, self.h)
        p_context = self.fully_connect(p_embedding)
        h_context = self.fully_connect(h_embedding)
        pos_result = self.cosine(p_context, h_context)
        neg_result = 1 - pos_result

        self.logits = tf.concat([pos_result, neg_result], axis=1)
        # 默认计算的最后有一个维度：softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
        self.prob = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.logits, axis=1)

    def fully_connect(self, x):
        x = tf.layers.dense(x, 256, activation='tanh')
        x = self.dropout(x)
        x = tf.layers.dense(x, 512, activation='tanh')
        x = self.dropout(x)
        x = tf.layers.dense(x, 256, activation='tanh')
        x = self.dropout(x)
        x = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2]))

        return x

    def create_loss(self):
        with tf.variable_scope("dssm_loss", reuse=tf.AUTO_REUSE):
            # 可以加入标签平滑策略进行优化
            y = tf.one_hot(self.y, self.calss_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
            self.loss = tf.reduce_mean(loss)

            # l2正则化
            # tf.trainable_variables()：可以也仅可以查看可训练的变量
            # tf.nn.l2_loss(tf_var) 返回 sum(tf_var**2 / 2)
            l2 = sum(1e-5 * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            self.loss += l2

            # 防止梯度爆炸
            # https://blog.csdn.net/cerisier/article/details/86523446
            tvars = tf.trainable_variables()
            # tf.gradients()接受求导值y和x
            # tf.clip_by_global_norm：梯度 * clip_norm / max(global_norm, clip_norm)
            # global_norm为所有梯度的平方和
            # https://blog.csdn.net/u013713117/article/details/56281715
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)

            self.train_op = RAdamOptimizer(self.learning_rate).minimize(self.loss)
            # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.y)
            # 准确度
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
