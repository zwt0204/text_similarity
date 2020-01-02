# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/12/25 11:06
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import os, json
import numpy as np
from model import SiameseLSTM
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.reset_default_graph()
import random


class SiameseTrainner:
    def __init__(self, is_training=True):
        self.class_graph = tf.Graph()
        self.model_dir = "siamese"
        self.batch_size = 128
        self.rnn_size = 256
        self.layer_size = 2
        self.sequence_length = 70
        self.grad_clip = 5
        self.learning_rate = 0.0003
        self.is_training = is_training
        self.vocab_file = "vocab.json"
        self.char_index = {' ': 0}
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.vocab_size = len(self.char_index) + 1
        if self.is_training is True:
            self.keep_prob = 0.3
        else:
            self.keep_prob = 1.0
        with tf.variable_scope('siamese_classification_query'):
            self.model = SiameseLSTM(self.rnn_size, self.layer_size, self.vocab_size, self.sequence_length,
                                     self.learning_rate, self.keep_prob, self.grad_clip)
        self.saver = tf.train.Saver()

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def train(self, epochs=30):
        p, h, y = self.load_char_data('train.txt')
        dataset = tf.data.Dataset.from_tensor_slices((self.model.input_x1, self.model.input_x2, self.model.y_data))
        dataset = dataset.batch(self.batch_size).repeat(epochs)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        with tf.Session()as sess:
            sess.run(tf.global_variables_initializer())
            if random.random() > 0.5:
                feed = {self.model.input_x1: p, self.model.input_x2: h, self.model.y_data: y}
            else:
                feed = {self.model.input_x1: p, self.model.input_x2: h, self.model.y_data: y}
            sess.run(iterator.initializer, feed_dict=feed)
            steps = int(len(y) / self.batch_size)
            for epoch in range(epochs):
                train_loss_value = 0.
                for step in range(steps):
                    p_batch, h_batch, y_batch = sess.run(next_element)
                    loss, _ = sess.run([self.model.cost, self.model.train_op],
                                       feed_dict={self.model.input_x1: p_batch,
                                                  self.model.input_x2: h_batch,
                                                  self.model.y_data: y_batch})
                    train_loss_value += loss / steps
                    if step % 100 == 0:
                        print('epoch:', epoch, ' step:', step, ' loss: ', train_loss_value)
                print("Epoch: %d/%d , train cost=%f" % ((epoch + 1), epochs, loss))
                self.saver.save(sess, os.path.join(self.model_dir, "sianesebirnn.dat"))

    # 加载char_index训练数据
    def load_char_data(self, file):
        path = file
        p = []
        h = []
        label = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                if json.loads(line)['similar'] == '':
                    continue
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
            if len(input_text) > self.sequence_length:
                count = self.sequence_length
            else:
                count = len(input_text)
            row = np.zeros((self.sequence_length), dtype=np.int32)
            for i in range(count):
                if input_text[i] in self.char_index.keys():
                    row[i] = self.char_index[input_text[i]]
                else:
                    row[i] = self.unknow_char_id
            result.append(row)
        return result


if __name__ == "__main__":
    trainner = SiameseTrainner(is_training=True)
    trainner.train()
