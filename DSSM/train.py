# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/12/20 18:07
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import os
import json
import tensorflow as tf
from model import DssmCore
import random
from utils.utils import data_util
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DssmTrainner:

    def __init__(self, vocab_file="vocab.json"):
        self.model_dir = "dssm"
        self.vocab_file = vocab_file
        self.char_index = {' ': 0}
        self.char_index = data_util.load_dict(vocab_file, self.char_index)
        self.unknow_char_id = len(self.char_index)
        self.io_sequence_size = 70
        vocab_size = len(self.char_index) + 1
        learning_rate = 0.0001
        trainable = True
        class_size = 2
        self.batch_size = 128
        self.keep_prob = 0.9
        with tf.variable_scope('dssm_query'):
            self.model = DssmCore(self.io_sequence_size, vocab_size, class_size, learning_rate, trainable)

    def train(self, epoch=30):
        p, h, y = self.load_char_data('train.txt')
        p_holder = tf.placeholder(dtype=tf.int32, shape=(None, self.io_sequence_size), name='p')
        h_holder = tf.placeholder(dtype=tf.int32, shape=(None, self.io_sequence_size), name='h')
        y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

        dataset = tf.data.Dataset.from_tensor_slices((p_holder, h_holder, y_holder))
        dataset = dataset.batch(self.batch_size).repeat(epoch)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        saver = tf.train.Saver()
        with tf.Session()as sess:
            sess.run(tf.global_variables_initializer())

            if random.random() > 0.5:
                feed = {p_holder: p, h_holder: h, y_holder: y}
            else:
                feed = {p_holder: h, h_holder: p, y_holder: y}
            sess.run(iterator.initializer, feed_dict=feed)
            steps = int(len(y) / self.batch_size)
            max_acc = 0.
            for epoch in range(epoch):
                for step in range(steps):
                    p_batch, h_batch, y_batch = sess.run(next_element)
                    _, loss, acc = sess.run([self.model.train_op, self.model.loss, self.model.acc],
                                            feed_dict={self.model.p: p_batch,
                                                       self.model.h: h_batch,
                                                       self.model.y: y_batch,
                                                       self.model.keep_prob: self.keep_prob})
                    print('epoch:', epoch, ' step:', step, ' loss: ', loss, ' acc:', acc)
                if acc >= max_acc:
                    max_acc = acc
                    saver.save(sess, f'dssm_{epoch}.ckpt')

    # 加载char_index训练数据
    def load_char_data(self, file):
        path = file
        p = []
        h = []
        label = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                p.append(json.loads(line)['question'])
                h.append(json.loads(line)['similar'])
                label.append(json.loads(line)['label'])

        p, h, label = data_util.shuffle(p, h, label)
        p_c_index = data_util.text_to_sequence(p, self.io_sequence_size, self.char_index, self.unknow_char_id)
        h_c_index = data_util.text_to_sequence(h, self.io_sequence_size, self.char_index, self.unknow_char_id)
        return p_c_index, h_c_index, label


if __name__ == "__main__":
    trainner = DssmTrainner()
    trainner.train()
