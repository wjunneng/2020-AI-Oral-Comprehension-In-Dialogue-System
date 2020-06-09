# -*- coding -*-
import os
import sys
import pathlib

project_dir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)

sys.path.append(project_dir)
os.chdir(sys.path[0])

import random
import numpy as np
import pandas as pd
from transformers import tokenization_bert
from src.core.predict import predict_sample

random.seed(42)
np.random.seed(42)


class Util(object):
    def __init__(self, model_type='albert'):
        self.model_type = model_type
        self.input_dir = os.path.join(project_dir, 'data', 'input')
        self.output_dir = os.path.join(project_dir, 'data', 'output')
        self.yanxishe_dir = os.path.join(project_dir, 'data', 'yanxishe')

        if not os.path.exists(self.yanxishe_dir):
            os.makedirs(self.yanxishe_dir)

        vocabs = []
        with open(os.path.join(project_dir, 'data', 'pre_train_model', 'chinese_wwm_pytorch', 'vocab.txt')) as file:
            for vocab in file.readlines():
                vocabs.append(vocab.strip('\n'))

        self.basic_tokenizer = tokenization_bert.BasicTokenizer(do_lower_case=True, never_split=None,
                                                                tokenize_chinese_chars=True)
        self.wordpiece_tokenizer = tokenization_bert.WordpieceTokenizer(vocab=vocabs, unk_token='[UNK]')

    def tokenization(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text=text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def write_data(self, df, label_file, seq_in_file, seq_out_file):
        # 'session_id', 'query', 'intent', 'slot_annotation'
        for index in range(df.shape[0]):
            query = df.iloc[index, 1]
            intent = df.iloc[index, 2]
            slot_annotation = df.iloc[index, 3]

            # intent
            label_file.write(intent + '\n')

            # query
            split_tokens = self.tokenization(text=query)

            # slot_annotation
            slot_median = ''
            start = slot_annotation.find('>')
            end = slot_annotation.find('</')
            if start != -1 and end != -1:
                slot_median = slot_annotation[start + 1:end]
            if slot_median == '':
                seq_out_file.write(' '.join(['O'] * len(split_tokens)) + '\n')
                seq_in_file.write(' '.join(split_tokens) + '\n')
            else:
                # TODO: 去除 || 之前的元素
                if '|' in slot_median and slot_median.count('|') == 2:
                    slot_median = slot_median[:slot_median.find('|')]

                query_index = query.find(slot_median)
                before = query[:query_index]
                after = query[query_index + len(slot_median):]

                before = self.tokenization(text=before)
                median = self.tokenization(text=slot_median)
                after = self.tokenization(text=after)

                slot_annotation_list = ['O'] * len(before)
                for index, value in enumerate(median):
                    annotation = slot_annotation[slot_annotation.find('<') + 1:slot_annotation.find('>')]
                    if index == 0:
                        slot_annotation_list.append('B-' + annotation)
                    else:
                        slot_annotation_list.append('I-' + annotation)
                slot_annotation_list += ['O'] * len(after)

                assert len(before + median + after) == len(slot_annotation_list)
                seq_in_file.write(' '.join(before + median + after) + '\n')
                seq_out_file.write(' '.join(slot_annotation_list) + '\n')

    def generate_yanxishe_input_data(self):
        """
        生成标准输入数据集
        :return:
        """
        # original
        test_csv_path = os.path.join(self.input_dir, 'test.csv')
        train_csv_path = os.path.join(self.input_dir, 'train.csv')
        slot_dictionaries_dir = os.path.join(self.input_dir, 'slot_dictionaries')

        # generate
        train_dir = os.path.join(self.yanxishe_dir, 'train')
        dev_dir = os.path.join(self.yanxishe_dir, 'dev')
        test_dir = os.path.join(self.yanxishe_dir, 'test')

        if not os.path.exists(train_dir) and not os.path.exists(dev_dir) and not os.path.exists(test_dir):
            os.makedirs(train_dir)
            os.makedirs(dev_dir)
            os.makedirs(test_dir)

        intent_label_path = os.path.join(self.yanxishe_dir, 'intent_label.txt')
        slot_label_path = os.path.join(self.yanxishe_dir, 'slot_label.txt')

        """
        'session_id', 'query', 'intent', 'slot_annotation'
        """
        train_data = pd.read_csv(train_csv_path, encoding='utf-8')
        test_data = pd.read_csv(test_csv_path, encoding='utf-8')

        """
        'music.play', 'music.pause', 'music.prev', 'music.next'
        'navigation.navigation', 'navigation.open', 'navigation.start_navigation', 'navigation.cancel_navigation'
        'phone_call.make_a_phone_call', 'phone_call.cancel'
        'OTHERS'
        """
        # print(train_data['intent'].unique())
        with open(intent_label_path, 'w', encoding='utf-8') as file:
            for item in ['music.play', 'music.pause', 'music.prev', 'music.next',
                         'navigation.navigation', 'navigation.open', 'navigation.start_navigation',
                         'navigation.cancel_navigation',
                         'phone_call.make_a_phone_call', 'phone_call.cancel',
                         'OTHERS']:
                file.write(item + '\n')

        """
        'age'
        'custom_destination'
        'emotion'
        'instrument'
        'language'
        'scene'
        'singer'
        'song'
        'style'
        'theme'
        'toplist'

        # 无对应的slot dictionaries
        'destination', 'contact_name', 'origin', 'phone_num'
        """
        slot_label_set = set()
        for item in train_data['slot_annotation']:
            start = item.find('<')
            if start != -1:
                slot_label_set.add(item[start + 1: item.find('>')])

        with open(slot_label_path, encoding='utf-8', mode='w') as file:
            for item in ['PAD', 'UNK', 'O']:
                file.write(item + '\n')

            for item in slot_label_set:
                file.write('B-' + item + '\n')
                file.write('I-' + item + '\n')

        train_rate = 0.8
        session_id_array = np.asarray(train_data['session_id'].unique())
        np.random.shuffle(session_id_array)
        train_id_array, dev_id_array = session_id_array[:int(len(session_id_array) * train_rate)], session_id_array[int(
            len(session_id_array) * train_rate):]

        train_df = pd.DataFrame()
        dev_df = pd.DataFrame()
        for session_id, id_data in train_data.groupby(by=['session_id']):
            if session_id in train_id_array:
                train_df = pd.concat([train_df, id_data], axis=0, ignore_index=True)
            else:
                dev_df = pd.concat([dev_df, id_data], axis=0, ignore_index=True)

        with open(os.path.join(train_dir, 'seq.in'), encoding='utf-8', mode='w') as train_seq_in_file, \
                open(os.path.join(train_dir, 'seq.out'), encoding='utf-8', mode='w') as train_seq_out_file, \
                open(os.path.join(train_dir, 'label'), encoding='utf-8', mode='w') as train_label_file, \
                open(os.path.join(dev_dir, 'seq.in'), encoding='utf-8', mode='w') as dev_seq_in_file, \
                open(os.path.join(dev_dir, 'seq.out'), encoding='utf-8', mode='w') as dev_seq_out_file, \
                open(os.path.join(dev_dir, 'label'), encoding='utf-8', mode='w') as dev_label_file:
            self.write_data(df=train_df, label_file=train_label_file, seq_in_file=train_seq_in_file,
                            seq_out_file=train_seq_out_file)
            self.write_data(df=dev_df, label_file=dev_label_file, seq_in_file=dev_seq_in_file,
                            seq_out_file=dev_seq_out_file)

        with open(os.path.join(test_dir, 'sample_pred_in.txt'), encoding='utf-8', mode='w') as file:
            for index in range(test_data.shape[0]):
                query = test_data.iloc[index, 1]
                split_tokens = self.tokenization(text=query)

                file.write(' '.join(split_tokens) + '\n')

    def select_token_from_slot_dictionary(self, text, token):
        slot_dictionaries_dir = os.path.join(self.input_dir, 'slot-dictionaries')

        for slot in ['age', 'custom_destination', 'emotion', 'instrument', 'language', 'scene', 'singer', 'song',
                     'style', 'theme', 'toplist']:
            slot_path = os.path.join(slot_dictionaries_dir, slot + '.txt')
            slot_token_list = []
            with open(file=slot_path, encoding='utf-8', mode='r') as file:
                for line in file.readlines():
                    line = line.strip().strip('\n')
                    slot_token_list.append(line)

            if text in slot_token_list:
                return slot

        return token

    def generate_result(self):
        test_dir = os.path.join(self.yanxishe_dir, 'test')
        sample_pred_out_txt_path = os.path.join(test_dir, 'sample_pred_out.txt')

        test_data = pd.read_csv(filepath_or_buffer=os.path.join(self.input_dir, 'test.csv'), encoding='utf-8')
        intent_list = []
        slot_annotation_list = []

        with open(sample_pred_out_txt_path, encoding='utf-8', mode='r') as file:
            for index, line in enumerate(file.readlines()):
                line = line.strip().strip('\n')
                line_list = line.split('\t')

                intent = line_list[0]
                slot_annotation = line_list[1]

                intent_list.append(intent)
                if intent == 'OTHERS':
                    slot_annotation_list.append(test_data.iloc[index, 1])
                else:
                    start = slot_annotation.find('<')
                    end = slot_annotation.find('>')
                    if start != -1 and end != -1:
                        token = slot_annotation[start + 1: end]

                        if '/' in token:
                            result = predict_sample(lines=[self.tokenization(test_data.iloc[index, 1])])
                            if result is not None:
                                slot_annotation = result

                        end_token = '</' + token + '>'
                        count = slot_annotation.count(end_token) - 1
                        if count != 0:
                            slot_annotation = slot_annotation.replace(end_token, '', count)

                        slot_annotation = slot_annotation.replace(' ', '')
                        slot_annotation = slot_annotation.replace('##', '')

                        if slot_annotation.find(end_token) == -1 and '//' not in end_token:
                            slot_annotation += end_token

                        token = token.replace(' ', '')
                        text = slot_annotation[
                               slot_annotation.find(token) + len(token) + 1:slot_annotation.rfind(token) - 2]

                        if text != '':
                            new_token = self.select_token_from_slot_dictionary(token=token, text=text)

                            slot_annotation = slot_annotation.replace(token, new_token)
                    else:
                        slot_annotation = slot_annotation.replace(' ', '')
                        slot_annotation = slot_annotation.replace('##', '')

                    slot_annotation_list.append(slot_annotation)

        test_data['intent'] = intent_list
        test_data['slot_annotation'] = slot_annotation_list

        test_data.to_csv('../../data/result.csv', encoding='utf-8', header=None, index=None)


if __name__ == '__main__':
    util = Util()
    # 生成标准的输入数据
    # util.generate_yanxishe_input_data()
    # 生成结果数据
    util.generate_result()
