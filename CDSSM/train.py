# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/12/30 17:01
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import os
import numpy as np
import json
import random
import tensorflow as tf
from model import CDSSM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CDssmTrainner:

    def __init__(self, vocab_file="vocab.json"):
        self.model_dir = "cdssm"
        self.vocab_file = vocab_file
        self.char_index = {' ': 0}
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.io_sequence_size = 70
        vocab_size = len(self.char_index) + 1
        self.num_lstm_units = 128
        self.batch_size = 32
        self.keep_prob = 0.4
        with tf.variable_scope('Cdssm_query'):
            self.model = CDSSM(self.vocab_file, self.num_lstm_units, vocab_size)

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def train(self, epochs=20):
        p, h, y = self.load_char_data('train.txt')
        p_holder = tf.placeholder(dtype=tf.int32, shape=(None, self.io_sequence_size), name='p')
        h_holder = tf.placeholder(dtype=tf.int32, shape=(None, self.io_sequence_size), name='h')
        y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')
        dataset = tf.data.Dataset.from_tensor_slices((p_holder, h_holder, y_holder))
        dataset = dataset.batch(self.batch_size).repeat(epochs)
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
            max_acc = 0
            for epoch in range(epochs):
                for step in range(steps):
                    p_batch, h_batch, y_batch = sess.run(next_element)

                    loss, _, acc = sess.run([self.model.loss, self.model.train_op, self.model.acc],
                                            feed_dict={self.model.queries: p_batch,
                                                       self.model.docs: h_batch,
                                                       self.model.y_data: y_batch,
                                                       self.model.sequence: self.io_sequence_size,
                                                       self.model.keep_prob: self.keep_prob})
                    print('epoch:', epoch, ' step:', step, ' loss: ', loss, 'acc', acc)
                if acc >= max_acc:
                    max_acc = acc
                    saver.save(sess, os.path.join(self.model_dir, "cdssm.dat"))

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

        p, h, label = self.shuffle(p, h, label)
        p_c_index = self.text_to_sequence(p[0:70016])
        h_c_index = self.text_to_sequence(h[0:70016])
        return p_c_index, h_c_index, label[0:70016]

    def shuffle(self, *arrs):
        arrs = list(arrs)
        for i, arr in enumerate(arrs):
            assert len(arrs[0]) == len(arrs[i])
            arrs[i] = np.array(arr)
        p = np.random.permutation(len(arrs[0]))
        return tuple(arr[p] for arr in arrs)

    def text_to_sequence(self, lines):
        result = []
        for input_text in lines:
            if len(input_text) > self.io_sequence_size:
                count = self.io_sequence_size
            else:
                count = len(input_text)
            row = np.zeros((self.io_sequence_size), dtype=np.int32)
            for i in range(count):
                if input_text[i] in self.char_index.keys():
                    row[i] = self.char_index[input_text[i]]
                else:
                    row[i] = self.unknow_char_id
            result.append(row)
        return result


if __name__ == "__main__":
    trainner = CDssmTrainner()
    trainner.train()