# -*- coding -*-
import os
import sys
import pathlib

project_dir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)

sys.path.append(project_dir)
os.chdir(sys.path[0])

import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

import random
import numpy as np
import pandas as pd
from transformers import tokenization_bert
from src.core.predict import predict_sample

random.seed(42)
np.random.seed(42)

input_dir = os.path.join(project_dir, 'data', 'input')
yanxishe_dir = os.path.join(project_dir, 'data', 'yanxishe')

if not os.path.exists(yanxishe_dir):
    os.makedirs(yanxishe_dir)


class Util(object):
    def __init__(self, model_type='albert'):
        self.model_type = model_type

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
        test_csv_path = os.path.join(input_dir, 'test.csv')
        train_csv_path = os.path.join(input_dir, 'train.csv')
        slot_dictionaries_dir = os.path.join(input_dir, 'slot_dictionaries')

        # generate
        train_dir = os.path.join(yanxishe_dir, 'train')
        dev_dir = os.path.join(yanxishe_dir, 'dev')
        test_dir = os.path.join(yanxishe_dir, 'test')

        if not os.path.exists(train_dir) and not os.path.exists(dev_dir) and not os.path.exists(test_dir):
            os.makedirs(train_dir)
            os.makedirs(dev_dir)
            os.makedirs(test_dir)

        intent_label_path = os.path.join(yanxishe_dir, 'intent_label.txt')
        slot_label_path = os.path.join(yanxishe_dir, 'slot_label.txt')

        """
        'session_id', 'query', 'intent', 'slot_annotation'
        """
        train_data = pd.read_csv(train_csv_path, encoding='utf-8')
        test_data = pd.read_csv(test_csv_path, encoding='utf-8')

        # digit_list = []
        # for index in range(train_data.shape[0]):
        #     if str(train_data.iloc[index, 1]).isdigit():
        #         digit_list.append(train_data.iloc[index, 0])
        #
        # digit_df = pd.DataFrame()
        # for session_id, id_data in train_data.groupby(by=['session_id']):
        #     if session_id in digit_list:
        #         digit_df = pd.concat([digit_df, id_data], axis=0, ignore_index=True)
        #
        # digit_df.to_csv('digit.csv', index=None)
        # print(digit_df)

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

        if not os.path.exists(slot_label_path):
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

    def generate_result(self):
        test_dir = os.path.join(yanxishe_dir, 'test')
        sample_pred_out_txt_path = os.path.join(test_dir, 'sample_pred_out.txt')

        test_data = pd.read_csv(filepath_or_buffer=os.path.join(input_dir, 'test.csv'), encoding='utf-8')
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

                        # [我要听大壮</singer>不一样] -> [我要听<singer>大壮</singer>不一样]
                        if '/' in token:
                            result = predict_sample(lines=[self.tokenization(test_data.iloc[index, 1])])
                            if result is not None:
                                slot_annotation = result
                            else:
                                # TODO: 播 放 歌 手 </song> 带 </song> 你 </song> 去 </song> 旅 </song> 游 </song>
                                slot_annotation = slot_annotation.replace('<' + token + '>', '')

                        # [<singer>张靓</singer>玫</singer>] -> [<singer>张靓玫</singer>]
                        end_token = '</' + token + '>'
                        count = slot_annotation.count(end_token) - 1
                        if count != 0:
                            slot_annotation = slot_annotation.replace(end_token, '', count)

                        slot_annotation = slot_annotation.replace(' ', '')
                        slot_annotation = slot_annotation.replace('##', '')

                        # [回<custom_destination>家] -> [回<custom_destination>家</custom_destination>]
                        if slot_annotation.find(end_token) == -1 and '//' not in end_token:
                            slot_annotation += end_token

                        # TODO: 不要直接替换, 在槽值不属于意图的情况下, 再进行替换
                        # token = token.replace(' ', '')
                        # text = slot_annotation[
                        #        slot_annotation.find(token) + len(token) + 1:slot_annotation.rfind(token) - 2]
                        # if text != '':
                        #     new_token = Slot.select_token_from_slot_dictionary(
                        #         slot_dir=os.path.join(input_dir, 'new-slot-dictionaries'), token=token, text=text, intent=intent)
                        #
                        #     slot_annotation = slot_annotation.replace(token, new_token)
                    else:
                        slot_annotation = slot_annotation.replace(' ', '')
                        slot_annotation = slot_annotation.replace('##', '')

                    slot_annotation_list.append(slot_annotation)

        test_data['intent'] = intent_list
        test_data['slot_annotation'] = slot_annotation_list

        test_data.to_csv('../../data/result.csv', encoding='utf-8', header=None, index=None)


class Slot(object):
    @staticmethod
    def select_token_from_slot_dictionary(slot_dir, text, intent, token):
        if 'new' in slot_dir:
            slot_dict = {'navigation.navigation': ['destination', 'custom_destination', 'origin'],
                         'phone_call.make_a_phone_call': ['phone_num', 'contact_name'],
                         'music.play': ['age', 'emotion', 'instrument', 'language', 'scene',
                                        'singer', 'song', 'style', 'theme', 'toplist']}
        else:
            slot_dict = {'music.play': ['age', 'emotion', 'instrument', 'language', 'scene',
                                        'singer', 'song', 'style', 'theme', 'toplist']}

        try:
            for slot in slot_dict[intent]:
                slot_path = os.path.join(slot_dir, slot + '.txt')
                slot_token_list = []
                with open(file=slot_path, encoding='utf-8', mode='r') as file:
                    for line in file.readlines():
                        line = line.strip().strip('\n')
                        slot_token_list.append(line)

                if text in slot_token_list:
                    token = slot
        except:
            return token
        finally:
            return token

    @staticmethod
    def generate_new_slot_dictionary(slot_dir, new_slot_dir):
        if not os.path.exists(new_slot_dir):
            os.makedirs(new_slot_dir)

        slot_list = ['age', 'custom_destination', 'emotion', 'instrument', 'language', 'scene', 'singer', 'song',
                     'style', 'theme', 'toplist']

        slot_dict = dict(zip(slot_list, [[]] * len(slot_list)))

        for slot_item in slot_list:
            slot_item_path = os.path.join(slot_dir, slot_item + '.txt')
            slot_item_list = []

            with open(slot_item_path, encoding='utf-8', mode='r') as file:
                for line in file.readlines():
                    line = line.strip().strip('\n')
                    slot_item_list.append(line)

            slot_dict[slot_item] = slot_item_list

        """
        'session_id', 'query', 'intent', 'slot_annotation'
        """
        train_csv_path = os.path.join(input_dir, 'train.csv')
        train_data = pd.read_csv(train_csv_path, encoding='utf-8')

        error_pair = []
        for index in range(train_data.shape[0]):
            session_id = train_data.iloc[index, 0]
            query = train_data.iloc[index, 1]
            intent = train_data.iloc[index, 2]
            slot_annotation = train_data.iloc[index, 3]

            if slot_annotation.find('<') != -1 and slot_annotation.find('>') != -1:
                token = slot_annotation[slot_annotation.find('<') + 1:slot_annotation.find('>')]
                median_slot = slot_annotation[slot_annotation.find('>') + 1:slot_annotation.find('</')]

                if '||' in median_slot:
                    error_pair.append(token + '\t' + median_slot)
                    continue

                if token not in slot_dict:
                    slot_dict[token] = [median_slot]
                else:
                    slot_dict[token].append(median_slot)

        for slot_item in slot_dict:
            slot_item_path = os.path.join(new_slot_dir, slot_item + '.txt')
            with open(slot_item_path, encoding='utf-8', mode='w') as file:
                for item in set(slot_dict[slot_item]):
                    file.write(item + '\n')

        with open(os.path.join(new_slot_dir, 'error_pairs.txt'), 'w', encoding='utf-8') as file:
            for item in error_pair:
                file.write(item + '\n')


class Rule(object):
    @staticmethod
    def rule():
        result_path = '../../data/result.csv'
        result = pd.read_csv(filepath_or_buffer=result_path, encoding='utf-8', header=None)
        result.columns = ['session_id', 'query', 'intent', 'slot_annotation']

        result = Rule.rule_1(result=result)

        result = Rule.rule_2(result=result)

        result = Rule.rule_4(result=result)

        result = Rule.rule_8(result=result)

        result.to_csv('../../data/result_rule.csv', encoding='utf-8', header=None, index=None)

    @staticmethod
    def rule_1(result: pd.DataFrame):
        """
        1、意图设计中未给出的语义槽，不用标注语义槽，但是需要标注意图。
        :param result:
        :return:
        """
        for index in range(result.shape[0]):
            session_id = int(result.iloc[index, 0])
            query = result.iloc[index, 1].strip()
            # 意图
            intent = result.iloc[index, 2].strip()
            slot_annotation = result.iloc[index, 3].strip()

            if slot_annotation.find('<') != -1 and slot_annotation.find('>') != -1 and slot_annotation.find('</') != -1:
                token = slot_annotation[slot_annotation.find('<') + 1:slot_annotation.find('>')]
                text = slot_annotation[slot_annotation.find('>') + 1:slot_annotation.find('</')]

                new_token = Slot.select_token_from_slot_dictionary(
                    slot_dir=os.path.join(project_dir, 'data', 'input', 'new-slot-dictionaries'), text=text,
                    intent=intent,
                    token=token)

                if new_token != token:
                    result.iloc[index, 3] = slot_annotation.replace(token, new_token)

        return result

    @staticmethod
    def rule_2(result: pd.DataFrame):
        """
        2、标注依赖于上文（不能使用下文），不是仅看当前 query。例如：“取消”，当最近的上文是 navigation 时，标注为
        navigation.cancel_navigation；当最近的上文是 music 时，标注为 music.pause；当最近的上文是 phone_call 时，
        标注为 phone_call.cancel；当位于 session 初始，没有明确领域信息时，标注为 OTHERS。
        :param result:
        :return:
        """
        # 'session_id', 'query', 'intent', 'slot_annotation'
        cancel_txt_path = os.path.join(project_dir, 'data', 'input', 'new-slot-dictionaries', 'cancel.txt')
        if not os.path.exists(cancel_txt_path):
            train_data = pd.read_csv(os.path.join(input_dir, 'train.csv'))

            navigation_cancel_navigation = train_data[train_data['intent'] == 'navigation.cancel_navigation']
            music_pause = train_data[train_data['intent'] == 'music.pause']
            phone_call_cancel = train_data[train_data['intent'] == 'phone_call.cancel']
            left_data = pd.concat([navigation_cancel_navigation, music_pause, phone_call_cancel], axis=0)

            with open(cancel_txt_path, mode='w') as file:
                for item in list(set(left_data['query'])):
                    write = True
                    for token in ['音乐', '开车', '导航', '歌', '曲', '电话', '地图', '行程', '通话']:
                        if token in item:
                            write = False
                    if write:
                        file.write(item + '\n')

        cancel_data = []
        with open(cancel_txt_path, mode='r') as file:
            for line in file.readlines():
                line = line.strip().strip('\n')
                cancel_data.append(line)

        cancel_session_id_list = []
        for index in range(result.shape[0]):
            session_id = int(result.iloc[index, 0])
            query = result.iloc[index, 1].strip()

            if query in cancel_data:
                cancel_session_id_list.append(session_id)

        before_id = None

        navigation_cancel_navigation = False
        music_pause = False
        phone_call_cancel = False
        other = True
        for index in range(result.shape[0]):
            session_id = int(result.iloc[index, 0])
            query = result.iloc[index, 1].strip()
            intent = result.iloc[index, 2].strip()
            slot_annotation = result.iloc[index, 3].strip()

            if session_id in cancel_session_id_list:
                if before_id != session_id:
                    navigation_cancel_navigation = False
                    music_pause = False
                    phone_call_cancel = False
                    other = True

                if 'navigation' in intent and query not in cancel_data:
                    other = False
                    music_pause = False
                    phone_call_cancel = False
                    navigation_cancel_navigation = True
                elif 'music' in intent and query not in cancel_data:
                    other = False
                    music_pause = True
                    phone_call_cancel = False
                    navigation_cancel_navigation = False
                elif 'phone_call' in intent and query not in cancel_data:
                    other = False
                    music_pause = False
                    phone_call_cancel = True
                    navigation_cancel_navigation = False

                if before_id is None or before_id != session_id:
                    if navigation_cancel_navigation:
                        other = False
                        music_pause = False
                        phone_call_cancel = False
                    elif music_pause:
                        other = False
                        phone_call_cancel = False
                        navigation_cancel_navigation = False
                    elif phone_call_cancel:
                        other = False
                        music_pause = False
                        navigation_cancel_navigation = False
                    else:
                        music_pause = False
                        phone_call_cancel = False
                        navigation_cancel_navigation = False

                    before_id = session_id

                    if query in cancel_data:
                        result.iloc[index, 2] = 'OTHERS'
                        continue

                if before_id == session_id and query in cancel_data:
                    if navigation_cancel_navigation:
                        result.iloc[index, 2] = 'navigation.cancel_navigation'
                    elif music_pause:
                        result.iloc[index, 2] = 'music.pause'
                    elif phone_call_cancel:
                        result.iloc[index, 2] = 'phone_call.cancel'
                    elif other:
                        result.iloc[index, 2] = 'OTHERS'

        return result

    @staticmethod
    def rule_3(result: pd.DataFrame):
        """
        3、单独的人名仅在上文为 phone_call 领域时，才标注为 phone_call. make_a_phone_call 意图，其余标注为 OTHERS。
        :param result:
        :return:
        """

    @staticmethod
    def rule_4(result: pd.DataFrame):
        """
        4、无上文影响的情况下，除了 11 位手机号码之外的纯数字 query，标注为 OTHERS。上文为 phone_call 时，3 位以上纯数字标注为 phone_call. make_a_phone_call。
        :param result:
        :return:
        """
        digit_list = []
        for index in range(result.shape[0]):
            if str(result.iloc[index, 1]).isdigit():
                result.iloc[index, 2] = 'OTHERS'
                result.iloc[index, 3] = result.iloc[index, 1]
                digit_list.append(int(result.iloc[index, 0]))

        before_id = None
        is_others = True
        for index in range(result.shape[0]):
            session_id = int(result.iloc[index, 0])
            query = result.iloc[index, 1].strip()
            intent = result.iloc[index, 2].strip()
            slot_annotation = result.iloc[index, 3].strip()

            if session_id in digit_list:
                if before_id is None or before_id != session_id:
                    is_others = True
                    before_id = session_id
                    if str(query).isdigit():
                        result.iloc[index, 2] = 'OTHERS'
                        result.iloc[index, 3] = query

                if session_id == before_id:
                    if str(query).isdigit() is False and 'phone_call.make_a_phone_call' == intent:
                        is_others = False

                    if str(query).isdigit() and is_others is False and len(query) > 3:
                        result.iloc[index, 2] = 'phone_call.make_a_phone_call'
                        result.iloc[index, 3] = '<phone_num>' + str(slot_annotation) + '</phone_num>'

        digit_df = pd.DataFrame()
        for session_id, id_data in result.groupby(by=['session_id']):
            if session_id in digit_list:
                digit_df = pd.concat([digit_df, id_data], axis=0, ignore_index=True)

        return result

    @staticmethod
    def rule_8(result: pd.DataFrame):
        """
        8、处理UNK
        :param result:
        :return:
        """
        for index in range(result.shape[0]):
            session_id = int(result.iloc[index, 0])
            query = result.iloc[index, 1].strip()
            intent = result.iloc[index, 2].strip()
            slot_annotation = result.iloc[index, 3].strip()

            if '[UNK]' in slot_annotation:
                sequence = ''
                unk_index = slot_annotation.find('[UNK]')

                for char_index in [-2, -1]:
                    if slot_annotation[unk_index + char_index] in query:
                        sequence += slot_annotation[unk_index + char_index]

                result.iloc[index, 3] = slot_annotation.replace('[UNK]', query[query.find(sequence) + len(sequence)])

            if ' ' in query and slot_annotation.find('<') != -1 and slot_annotation.find('>') != -1:
                space_indexs = list(set([query.find(' '), query.rfind(' ')]))

                for space_index in space_indexs:
                    match_sequence = query[space_index - 1] + query[space_index + 1]

                    match_index = slot_annotation.find(match_sequence)
                    if match_index != -1:
                        slot_annotation = slot_annotation[
                                          :match_index + len(match_sequence) - 1] + ' ' + slot_annotation[
                                                                                          match_index + len(
                                                                                              match_sequence) - 1:]

                result.iloc[index, 3] = slot_annotation

        return result


if __name__ == '__main__':
    util = Util()

    # 生成新的slot dictionary
    # Slot.generate_new_slot_dictionary(slot_dir=os.path.join(project_dir, 'data', 'input', 'slot-dictionaries'),
    #                                   new_slot_dir=os.path.join(project_dir, 'data', 'input', 'new-slot-dictionaries'))

    # 生成标准的输入数据
    # util.generate_yanxishe_input_data()

    # 生成结果数据
    # util.generate_result()

    # 经过规则
    Rule.rule()
