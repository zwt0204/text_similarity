# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/12/25 11:10
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf
import json
from utils.radam import RAdamOptimizer


class SimpleLSTMCell(tf.contrib.rnn.RNNCell):
    """
    The simpler version of LSTM cell with forget gate set to 1, according to the paper.
    """

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "simple_lstm_cell", reuse=self._reuse):
            c, h = state
            if not hasattr(self, '_wi'):
                self._wi = tf.get_variable('simple_lstm_cell_wi', dtype=tf.float32,
                                           shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self._num_units],
                                           initializer=tf.orthogonal_initializer())
                self._bi = tf.get_variable('simple_lstm_cell_bi', dtype=tf.float32, shape=[self._num_units],
                                           initializer=tf.constant_initializer(0.0))
                self._wo = tf.get_variable('simple_lstm_cell_wo', dtype=tf.float32,
                                           shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self._num_units],
                                           initializer=tf.orthogonal_initializer())
                self._bo = tf.get_variable('simple_lstm_cell_bo', dtype=tf.float32, shape=[self._num_units],
                                           initializer=tf.constant_initializer(0.0))
                self._wc = tf.get_variable('simple_lstm_cell_wc', dtype=tf.float32,
                                           shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self._num_units],
                                           initializer=tf.orthogonal_initializer())
                self._bc = tf.get_variable('simple_lstm_cell_bc', dtype=tf.float32, shape=[self._num_units],
                                           initializer=tf.constant_initializer(0.0))
            i = tf.nn.sigmoid(tf.matmul(tf.concat([inputs, h], 1), self._wi) + self._bi)
            o = tf.nn.sigmoid(tf.matmul(tf.concat([inputs, h], 1), self._wo) + self._bo)
            _c = self._activation(tf.matmul(tf.concat([inputs, h], 1), self._wc) + self._bc)
            # remove forget gate according to the paper
            new_c = c + i * _c
            new_h = o * self._activation(new_c)

            return new_h, (new_c, new_h)


class LSTMDSSM(object):
    """
    The LSTM-DSSM model refering to the paper: Deep Sentence Embedding Using Long Short-Term Memory Networks: Analysis and Application to Information Retrieval.
    papaer available at: https://arxiv.org/abs/1502.06922
    """

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
        self.vocab_size = vocab_size
        self.grad_clip = 5
        self.sequence = 70
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
        self.embed_docs = tf.nn.embedding_lookup(self.embed, self.docs)
        with tf.variable_scope('query_lstm'):
            self.cell_q = SimpleLSTMCell(self.num_lstm_units)
        with tf.variable_scope('doc_lstm'):
            self.cell_d = SimpleLSTMCell(self.num_lstm_units)

        # shape = [batch_size, sequence, hidden_size]
        outputs1, state = tf.nn.dynamic_rnn(self.cell_q, self.embed_queries, self.sequence, dtype=tf.float32,
                                            scope="simple_lstm_cell_query")
        outputs2, state = tf.nn.dynamic_rnn(self.cell_d, self.embed_docs, self.sequence, dtype=tf.float32,
                                            scope="simple_lstm_cell_doc")

        outputs1 = tf.reduce_mean(outputs1, axis=1)
        outputs2 = tf.reduce_mean(outputs2, axis=1)

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
            # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.y_data)
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.y_data, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32'), name='accuracy')
