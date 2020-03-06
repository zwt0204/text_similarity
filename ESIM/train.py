# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/12/25 11:15
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import os
import numpy as np
import json
import tensorflow as tf
from model import EsimCore
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class EsimTrainner:

    def __init__(self, vocab_file="D:\gitwork\qa_robot\\resource\\vocab\dictionary.json"):
        self.model_dir = "esim"
        self.vocab_file = vocab_file
        self.char_index = {' ': 0}
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.io_sequence_size = 50
        vocab_size = len(self.char_index) + 1
        learning_rate = 0.0003
        trainable = True
        class_size = 2
        self.batch_size = 64
        self.keep_prob = 0.3
        with tf.variable_scope('esim_query'):
            self.model = EsimCore(self.io_sequence_size, vocab_size, class_size, learning_rate, trainable)

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def train(self, epoch=20):
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
                    saver.save(sess, os.path.join(self.model_dir, "similarity.dat"))

    # 加载char_index训练数据
    def load_char_data(self, file):
        path = file
        p = []
        h = []
        label = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                data = json.loads(line)
                p.append(data['question'])
                h.append(data['similar'])
                label.append(data['label'])

        p, h, label = self.shuffle(p, h, label)
        p_c_index = self.text_to_sequence(p)
        h_c_index = self.text_to_sequence(h)
        return p_c_index, h_c_index, label

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
    trainner = EsimTrainner()
    trainner.train()
