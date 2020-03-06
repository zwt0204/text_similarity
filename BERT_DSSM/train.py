# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/12/26 16:16
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from model import DssmCore
from data_process import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DssmTrainner:

    def __init__(self):
        self.model_dir = "dssm"
        self.io_sequence_size = 70
        learning_rate = 0.0001
        trainable = True
        class_size = 2
        self.batch_size = 128
        self.keep_prob = 0.9
        self.label_list = TextProcessor().get_labels()
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.model.vocab_file, do_lower_case=False)
        with tf.variable_scope('dssm_query'):
            self.model = DssmCore(self.io_sequence_size, class_size, learning_rate, trainable)

    def feed_data(self, batch_ids, batch_mask, batch_segment, batch_label,
                  batch_ids_1, batch_mask_1, batch_segment_1, batch_label_1, keep_prob):
        """构建text_model需要传入的数据"""
        feed_dict = {
            self.model.input_ids: np.array(batch_ids),
            self.model.input_mask: np.array(batch_mask),
            self.model.segment_ids: np.array(batch_segment),
            self.model.labels: np.array(batch_label),
            self.model.input_ids_1: np.array(batch_ids_1),
            self.model.input_mask_1: np.array(batch_mask_1),
            self.model.segment_ids_1: np.array(batch_segment_1),
            self.model.labels_1: np.array(batch_label_1),
            self.model.keep_prob: keep_prob
        }
        return feed_dict

    def train(self, epochs=30):

        train_examples = TextProcessor().get_train_examples('data')
        trian_data = convert_examples_to_features(train_examples, self.label_list, self.io_sequence_size,
                                                  self.tokenizer)
        trian_data_1 = convert_examples_to_features(train_examples, self.label_list, self.io_sequence_size,
                                                  self.tokenizer)

        saver = tf.train.Saver()
        with tf.Session()as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                batch_train = batch_iter(trian_data, self.batch_size)
                batch_train_1 = batch_iter(trian_data_1, self.batch_size)
                for batch_ids, batch_ids_1, batch_mask, batch_mask_1, batch_segment, batch_segment_1, batch_label, batch_label_1 in zip(batch_train, batch_train_1):
                    feed_dict = self.feed_data(batch_ids, batch_mask, batch_segment, batch_label, batch_ids_1, batch_mask_1, batch_segment_1, batch_label_1, self.model.keep_prob)
                    train_loss, train_accuracy = sess.run(
                        [self.model.loss,
                         self.model.acc], feed_dict=feed_dict)
                    print(train_loss, train_accuracy)
                saver.save(sess, 'model')


if __name__ == "__main__":
    trainner = DssmTrainner()
    trainner.train()
