# -*- encoding: utf-8 -*-
"""
@File    : predict.py
@Time    : 2019/12/25 11:07
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import os, json
import numpy as np
from model import SiameseLSTM
import tensorflow as tf
import logging

_logger = logging.getLogger()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SiamesePredict:
    def __init__(self):
        self.model_dir = "siamese"
        self.vocab_file = "vocab.json"
        self.batch_size = 60
        self.rnn_size = 128
        self.layer_size = 2
        self.learning_rate = 0.0003
        self.sequence_length = 70
        self.grad_clip = 5
        self.char_index = {' ': 0}
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.vocab_size = len(self.char_index) + 1
        self.keep_prob = 1.0
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('siamese_classification_query'):
                self.model = SiameseLSTM(self.rnn_size, self.layer_size, self.vocab_size, self.sequence_length,
                                         self.learning_rate, self.keep_prob, self.grad_clip)
            self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        self.session = tf.Session(graph=self.graph, config=config)
        with self.session.as_default():
            self.load()

    def load(self):
        _logger.info('======================%s' % self.model_dir)
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        self.saver.restore(self.session, ckpt.model_checkpoint_path)
        logging.info("siamese load {} success...".format(str(ckpt.model_checkpoint_path)))

    def close(self):
        if self.session is not None:
            self.session.close()

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def predict(self, input_text, question_standard):
        input_text = str((input_text + ' ') * len(question_standard)).strip().split(' ')
        p, h = self.load_char_data_predict(input_text, question_standard)
        prediction, prob = self.session.run(
            [self.model.prediction, self.model.prob],
            feed_dict={self.model.input_x1: p,
                       self.model.input_x2: h})
        return prediction, prob

    def predict_batch(self, input_text, question_standard):
        input_text = str((input_text + ' ') * len(question_standard)).strip().split(' ')
        input_vector = []
        input_vector2 = []
        for i in range(len(input_text)):
            raw_vector = self.convert_vector(input_text[i], self.model.sequence_length)
            input_vector.append(raw_vector)
        input_vector = np.array(input_vector, dtype=np.int32)
        for i in range(len(question_standard)):
            raw_vector = self.convert_vector(question_standard[i], self.model.sequence_length)
            input_vector2.append(raw_vector)
        input_vector2 = np.array(input_vector2, dtype=np.int32)
        prediction, prob = self.session.run(
            [self.model.prediction, self.model.prob],
            feed_dict={self.model.input_x1: input_vector,
                       self.model.input_x2: input_vector2})
        return prediction, prob.tolist()

    def predict_batch_test(self, input_text, question_standard, batch_size):
        batch_count = int(len(input_text) / batch_size)
        result = []
        for i in range(batch_count):
            batch_xitems = input_text[i * batch_size:(i + 1) * batch_size]
            batch_yitems = question_standard[i * batch_size:(i + 1) * batch_size]
            input_vector, input_vector2 = self.convert_batch(batch_xitems, batch_yitems, batch_size)
            prediction, prob = self.session.run(
                [self.model.prediction, self.model.prob],
                feed_dict={self.model.input_x1: input_vector,
                           self.model.input_x2: input_vector2})
            result.extend(prob.tolist())
        length = len(result)
        data = []
        print(length)
        for i in range(0, length, 2):
            data.append(result[i:i + 2])

        return data
        # return result[0:length:2]

    # 加载char_index训练数据
    def load_char_data(self, file):
        path = file
        p = []
        h = []
        label = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                p.append(json.loads(line)['question'])
                h.append(json.loads(line)['syn'])
                label.append(json.loads(line)['label'])

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

    def convert_vector(self, input_text, limit):
        char_vector = np.zeros((self.model.sequence_length), dtype=np.int32)
        count = len(input_text.strip().lower())
        if count > limit:
            count = limit
        for i in range(count):
            if input_text[i] in self.char_index.keys():
                char_vector[i] = self.char_index[input_text[i]]
        return char_vector

    def load_char_data_predict(self, p, h):
        p_c_index = self.text_to_sequence(p)
        h_c_index = self.text_to_sequence(h)
        return p_c_index, h_c_index

    def convert_batch(self, xitems, yitems, batch_size):
        xrecords = np.zeros((batch_size, self.model.sequence_length))
        yrecords = np.zeros((batch_size, self.model.sequence_length))
        for i in range(len(xitems)):
            count = len(xitems[i])
            if count > self.model.sequence_length:
                count = self.model.sequence_length
            count2 = len(yitems[i])
            if count2 > self.model.sequence_length:
                count2 = self.model.sequence_length
            for j in range(count):
                if xitems[i][j] in self.char_index.keys():
                    xrecords[i][j] = self.char_index[xitems[i][j]]
            for k in range(count2):
                if yitems[i][k] in self.char_index.keys():
                    yrecords[i][k] = self.char_index[yitems[i][k]]
        return xrecords, yrecords


if __name__ == "__main__":
    SiameseT = SiamesePredict()
    SiameseT.predict('data1', 'data2')
