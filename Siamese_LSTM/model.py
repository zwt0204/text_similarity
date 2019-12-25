# -*- encoding: utf-8 -*-
"""
@File    : mdoel.py
@Time    : 2019/12/23 19:33
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf
from utils.radam import RAdamOptimizer


class SiameseLSTM(object):

    def __init__(self, rnn_size, layer_size, vocab_size, sequence_length, learning_rate=0.001, keep_prob=0.5,
                 grad_clip=5, trainable=False):
        self.is_training = trainable
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.embedding_size = 200
        self.hidden_size = 256
        self.output_class_size = 2
        self.rnn_size = 256
        self.layer_size = layer_size
        self.keep_prob = keep_prob
        self.grad_clip = grad_clip
        with tf.name_scope("siamese_declare"):
            self.input_x1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
            self.input_x2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_y')
            self.y_data = tf.placeholder(tf.float32, shape=[None], name='y_data')
            with tf.device('/cpu:0'):
                embedding = self.weight_variables([vocab_size, rnn_size], 'embedding')
                inputs_x1 = tf.nn.embedding_lookup(embedding, self.input_x1)
                inputs_x2 = tf.nn.embedding_lookup(embedding, self.input_x2)
            self.inputs_x1 = self.transform_inputs(inputs_x1, rnn_size, sequence_length)
            self.inputs_x2 = self.transform_inputs(inputs_x2, rnn_size, sequence_length)
        self.create_declare()
        self.build()
        self.create_loss()

    def create_declare(self):
        with tf.variable_scope('dense_layer'):
            self.fc_w1 = self.weight_variables([2 * self.rnn_size, 128], 'fc_w1')
            self.fc_w2 = self.weight_variables([2 * self.rnn_size, 128], 'fc_w2')
            self.fc_b1 = self.bias_variables([128], 'fc_b1')
            self.fc_b2 = self.bias_variables([128], 'fc_b2')

    def build(self):
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(self.rnn_size) for _ in range(self.layer_size)]
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                           output_keep_prob=self.keep_prob)
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(self.rnn_size) for _ in range(self.layer_size)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list),
                                                           output_keep_prob=self.keep_prob)
        # backward rnn
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            outputs_x1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, self.inputs_x1,
                                                                       dtype=tf.float32)
            output_x1 = tf.reduce_mean(outputs_x1, 0)
            # 开启变量重用的开关
            tf.get_variable_scope().reuse_variables()
            outputs_x2, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, self.inputs_x2,
                                                                       dtype=tf.float32)
            output_x2 = tf.reduce_mean(outputs_x2, 0)
            self.logits_1 = tf.matmul(output_x1, self.fc_w1) + self.fc_b1
            self.logits_2 = tf.matmul(output_x2, self.fc_w2) + self.fc_b2

        f_x1x2 = tf.reduce_sum(tf.multiply(self.logits_1, self.logits_2), 1)
        norm_fx1 = tf.sqrt(tf.reduce_sum(tf.square(self.logits_1), 1))
        norm_fx2 = tf.sqrt(tf.reduce_sum(tf.square(self.logits_2), 1))
        self.Ew = f_x1x2 / (norm_fx1 * norm_fx2)
        self.Bw = 1 - self.Ew
        self.logits = tf.concat([self.Ew, self.Bw], axis=0)
        self.prob = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.logits)

    def create_loss(self):
        with tf.name_scope("siamese_loss"):
            self.cost = self.contrastive_loss(self.Ew, self.y_data)

            # train optimization
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_clip)
            optimizer = RAdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def weight_variables(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name=name)

    def bias_variables(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    def transform_inputs(self, inputs, rnn_size, sequence_length):
        # 其中0表示三维数组的高也就是二维数组的个数，1表示二维数组的行、2表示二维数组的列
        # [1, 0, 2]表示将三维数组的高和行进行转置
        inputs = tf.transpose(inputs, [1, 0, 2])
        inputs = tf.reshape(inputs, [-1, rnn_size])
        # 在第0维度进行切割
        inputs = tf.split(inputs, sequence_length, 0)
        return inputs

    # 对比损失
    # http://note.youdao.com/noteshare?id=ed28b56c8599b0fbb8632e425df600fb&sub=B345BFE1D1A843279AB251B4183692B0
    def contrastive_loss(self, Ew, y):
        # tf.square对其中的每个元素求平方
        # 1 - Ew 余弦距离
        l_1 = y * 0.25 * tf.square(1 - Ew)
        l_0 = (1 - y) * tf.square(tf.maximum(Ew, 0))
        loss = tf.reduce_sum(l_1 + l_0)
        return loss
