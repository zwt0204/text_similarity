# -*- encoding: utf-8 -*-
"""
@File    : predict.py
@Time    : 2019/12/25 11:16
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import os
import numpy as np
import tensorflow as tf
from model import EsimCore

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import json

_logger = logging.getLogger()


class EsimPredicter:
    def __init__(self):
        self.model_dir = 'esim'
        self.vocab_file = 'vocab.json'
        self.char_index = {' ': 0}
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.io_sequence_size = 70
        self.vocab_size = len(self.char_index) + 1
        self.is_training = False
        self.batch_size = 60
        self.class_size = 2
        self.learning_rate = 0.0001
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('esim_query'):
                self.model = EsimCore(io_sequence_size=self.io_sequence_size, vocab_size=self.vocab_size,
                                      class_size=self.class_size, learning_rate=self.learning_rate,
                                      trainable=self.is_training)
            self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        self.session = tf.Session(graph=self.graph, config=config)
        with self.session.as_default():
            self.load()

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def load(self):
        _logger.info('======================%s' % self.model_dir)
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        self.saver.restore(self.session, ckpt.model_checkpoint_path)
        logging.info("esim load {} success...".format(str(ckpt.model_checkpoint_path)))

    def close(self):
        if self.session is not None:
            self.session.close()

    def convert_xrow(self, input_text):
        char_vector = np.zeros(self.io_sequence_size, dtype=np.int32)
        for i in range(len(input_text)):
            char_value = input_text[i]
            if char_value in self.char_index.keys():
                char_vector[i] = self.char_index[char_value]
        return char_vector

    def predict(self, input_text, question_standard):
        input_text = str((input_text + ' ') * len(question_standard)).strip().split(' ')
        p, h = self.load_char_data_predict(input_text, question_standard)
        pr, no = self.session.run(
            [self.model.prediction, self.model.prob],
            feed_dict={self.model.p: p,
                       self.model.h: h,
                       self.model.keep_prob: 1})
        score = []

        # for j in no.tolist():
        #     score.append(j[1])
        # return score

        for i, value in enumerate(no.tolist()):
            score.append(value[1])
        return pr, score

    def load_char_data_predict(self, p, h):
        p_c_index = self.text_to_sequence(p)
        h_c_index = self.text_to_sequence(h)
        return p_c_index, h_c_index

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

    def predict_batch_test(self, input_text, question_standard, batch_size):
        batch_count = int(len(input_text) / batch_size)
        result = []
        for i in range(batch_count):
            batch_xitems = input_text[i * batch_size:(i + 1) * batch_size]
            batch_yitems = question_standard[i * batch_size:(i + 1) * batch_size]
            input_vector, input_vector2 = self.convert_batch(batch_xitems, batch_yitems, batch_size)
            prediction, prob = self.session.run(
                [self.model.prediction, self.model.prob],
                feed_dict={self.model.p: input_vector,
                           self.model.h: input_vector2,
                           self.model.keep_prob: 1
                           })
            result.extend(prob.tolist())
        #     for j in prob.tolist():
        #         result.append(j[1])
        return result

    def convert_batch(self, xitems, yitems, batch_size):
        xrecords = np.zeros((batch_size, self.model.io_sequence_size))
        yrecords = np.zeros((batch_size, self.model.io_sequence_size))
        for i in range(len(xitems)):
            count = len(xitems[i])
            if count > self.model.io_sequence_size:
                count = self.model.io_sequence_size
            count2 = len(yitems[i])
            if count2 > self.model.io_sequence_size:
                count2 = self.model.io_sequence_size
            for j in range(count):
                if xitems[i][j] in self.char_index.keys():
                    xrecords[i][j] = self.char_index[xitems[i][j]]
            for k in range(count2):
                if yitems[i][k] in self.char_index.keys():
                    yrecords[i][k] = self.char_index[yitems[i][k]]
        return xrecords, yrecords


if __name__ == "__main__":
    MODLE_ESIM = EsimPredicter()
    MODLE_ESIM.predict('data1', 'data2')