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
from BERT_DSSM.al_bert import modeling

_logger = logging.getLogger()


class DssmCore:
    """相当于伪孪生网络，两个网络之间不共享参数，各自学习"""

    def __init__(self, io_sequence_size, class_size=2, learning_rate=0.001, trainable=False):
        # 为True表示训练
        self.is_training = trainable
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

        self.vocab_file = './bert_model/chinese_L-12_H-768_A-12/vocab.txt'  # the path of vocab file
        self.bert_config_file = './bert_model/chinese_L-12_H-768_A-12/albert_config.json'  # the path of bert_cofig file
        self.init_checkpoint = './bert_model/chinese_L-12_H-768_A-12/albert_model.ckpt'  # the path of bert model
        self.use_one_hot_embeddings = False
        with tf.variable_scope('dssm_declare', reuse=tf.AUTO_REUSE):

            self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
            self.input_ids = tf.placeholder(tf.int64, shape=[None, self.io_sequence_size], name='input_ids')
            self.input_mask = tf.placeholder(tf.int64, shape=[None, self.io_sequence_size], name='input_mask')
            self.segment_ids = tf.placeholder(tf.int64, shape=[None, self.io_sequence_size], name='segment_ids')
            self.labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')

            self.input_ids_1 = tf.placeholder(tf.int64, shape=[None, self.io_sequence_size], name='input_ids_1')
            self.input_mask_1 = tf.placeholder(tf.int64, shape=[None, self.io_sequence_size], name='input_mask_1')
            self.segment_ids_1 = tf.placeholder(tf.int64, shape=[None, self.io_sequence_size], name='segment_ids_1')
            self.labels_1 = tf.placeholder(tf.int64, shape=[None, ], name='labels_1')

        self.create_embedding()
        self.create_model()
        if self.is_training is True:
            self.create_loss()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def create_embedding(self):
        with tf.name_scope('bert'):
            bert_model = modeling.BertModel(
                config=self.bert_config,
                is_training=self.is_training,
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=self.use_one_hot_embeddings)

            bert_model_1 = modeling.BertModel(
                config=self.bert_config,
                is_training=self.is_training,
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=self.use_one_hot_embeddings)

            self.p_embedding = bert_model.get_pooled_output()
            self.h_embedding = bert_model_1.get_pooled_output()

    def cosine(self, p, h):
        # 用于计算向量，矩阵和tensor的范数，默认情况下是计算欧氏距离的L2范数
        # keep_dims是否保持维度不变
        p_norm = tf.norm(p, axis=1, keepdims=True)
        h_norm = tf.norm(p, axis=1, keepdims=True)
        # 公式：http://note.youdao.com/noteshare?id=b04ad8ab7211331073217e8202056585&sub=44E9A792C5D04903A73118503699E7D9
        cosine = tf.reduce_sum(tf.multiply(p, h), axis=1, keepdims=True) / (p_norm * h_norm)

        return cosine

    def create_model(self):
        # p_embedding = tf.nn.embedding_lookup(self.embedding, self.p)
        # h_embedding = tf.nn.embedding_lookup(self.embedding, self.h)
        p_context = self.fully_connect(self.p_embedding)
        h_context = self.fully_connect(self.h_embedding)
        # p_context:[batch_size, ]
        pos_result = self.cosine(p_context, h_context)
        neg_result = 1 - pos_result

        self.logits = tf.concat([pos_result, neg_result], axis=1)
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
            y = tf.one_hot(self.labels, self.calss_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
            self.loss = tf.reduce_mean(loss)

            # l2正则化
            # tf.trainable_variables()：可以也仅可以查看可训练的变量
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
            correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.labels)
            # 准确度
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
