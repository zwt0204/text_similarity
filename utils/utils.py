# -*- encoding: utf-8 -*-
"""
@File    : utils.py
@Time    : 2019/12/23 9:42
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import numpy as np
import json


class data_utils:

    def __init__(self):
        super(data_utils, self).__init__()

    def shuffle(self, *arrs):
        arrs = list(arrs)
        for i, arr in enumerate(arrs):
            assert len(arrs[0]) == len(arrs[i])
            arrs[i] = np.array(arr)
        p = np.random.permutation(len(arrs[0]))
        return tuple(arr[p] for arr in arrs)

    def text_to_sequence(self, lines, io_sequence_size, char_index, unknow_char_id):
        result = []
        for input_text in lines:
            if len(input_text) > io_sequence_size:
                count = io_sequence_size
            else:
                count = len(input_text)
            row = np.zeros(io_sequence_size, dtype=np.int32)
            for i in range(count):
                if input_text[i] in char_index.keys():
                    row[i] = char_index[input_text[i]]
                else:
                    row[i] = unknow_char_id
            result.append(row)
        return result

    def load_dict(self, vocab_file, char_index):
        i = 0
        with open(vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                char_index[charvalue.strip()] = i + 1
                i += 1
        return char_index


data_util = data_utils()