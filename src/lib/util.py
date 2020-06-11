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

import difflib
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


class Eda(object):
    def __init__(self):
        self.train_csv_path = os.path.join(input_dir, 'train.csv')
        self.test_csv_path = os.path.join(input_dir, 'test.csv')

    def match_train_test(self):
        train_data = pd.read_csv(self.train_csv_path, encoding='utf-8')
        test_data = pd.read_csv(self.test_csv_path, encoding='utf-8')

        train_list = []
        test_list = []
        for group_session_id, group_train_data in train_data.groupby(by=['session_id']):
            train_list.append(' '.join(group_train_data['query'].values))

        for group_session_id, group_test_data in test_data.groupby(by=['session_id']):
            test_list.append(' '.join(group_test_data['query'].values))

        print('\n')
        # train_list.length: 17010
        # test_list.length: 4342
        print('train_list.length: {}'.format(len(train_data)))
        print('test_list.length: {}'.format(len(test_data)))
        for test_sample in test_list:
            if test_sample in train_list:
                # 仅仅匹配一条数据： 我要回家 取消 取消
                print(test_sample)

        train_list = list(set(train_list))
        test_list = list(set(test_list))

        # train_list.length: 17010
        # test_list.length: 4342
        print('train_list.length: {}'.format(len(train_data)))
        print('test_list.length: {}'.format(len(test_data)))


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
                            token = token.replace('/', '')
                            result = predict_sample(lines=[self.tokenization(test_data.iloc[index, 1])])
                            if result is not None:
                                slot_annotation = result
                            else:
                                # TODO: 播 放 歌 手 </song> 带 </song> 你 </song> 去 </song> 旅 </song> 游 </song>
                                slot_annotation = slot_annotation.replace('<' + token + '>', '')

                        # [<singer>张靓</singer>玫</singer>] -> [<singer>张靓玫</singer>]
                        end_token = '</' + token + '>'
                        count = slot_annotation.count(end_token) - 1

                        if '<' + token + '>' not in slot_annotation:
                            count = slot_annotation.count(end_token)

                        if count != 0:
                            slot_annotation = slot_annotation.replace(end_token, '', count)

                        slot_annotation = slot_annotation.replace(' ', '')
                        slot_annotation = slot_annotation.replace('##', '')

                        # [回<custom_destination>家] -> [回<custom_destination>家</custom_destination>]
                        if slot_annotation.find(
                                end_token) == -1 and '//' not in end_token and '<' + token + '>' in slot_annotation:
                            slot_annotation += end_token

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
    def filter_slot(slot_item_set_list):
        """
        过滤无必要的singer和song
        :param slot_item_set_list:
        :return:
        """
        slot_item_set_list = [i for i in slot_item_set_list if len(i) > 1]

        # slot_item_set_list = sorted(slot_item_set_list, key=lambda a: len(a))
        # result = []
        # for i in range(len(slot_item_set_list)):
        #     to_add = True
        #     for j in range(i + 1, len(slot_item_set_list)):
        #         if slot_item_set_list[i] in slot_item_set_list[j]:
        #             to_add = False
        #             j_copy = j
        #     if to_add:
        #         result.append(slot_item_set_list[i])
        #     else:
        #         print(slot_item_set_list[i], slot_item_set_list[j_copy])

        return slot_item_set_list

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
                slot_item_set_list = set(slot_dict[slot_item])
                if slot_item in ['singer', 'song']:
                    slot_item_set_list = Slot.filter_slot(slot_item_set_list)

                for item in slot_item_set_list:
                    file.write(item + '\n')

        with open(os.path.join(new_slot_dir, 'error_pairs.txt'), 'w', encoding='utf-8') as file:
            for item in set(error_pair):
                file.write(item + '\n')


class Rule(object):
    @staticmethod
    def rule():
        result_path = '../../data/result.csv'
        result = pd.read_csv(filepath_or_buffer=result_path, encoding='utf-8', header=None)
        result.columns = ['session_id', 'query', 'intent', 'slot_annotation']

        result = Rule.rule_0(result=result)

        result = Rule.rule_1(result=result)

        result = Rule.rule_2(result=result)

        result = Rule.rule_3(result=result)

        result = Rule.rule_4(result=result)

        result = Rule.rule_8(result=result)

        result.to_csv('../../data/result_rule.csv', encoding='utf-8', header=None, index=None)

    @staticmethod
    def rule_0(result: pd.DataFrame):
        """
        0、处理UNK 和 空格
        :param result:
        :return:
        """
        for index in range(result.shape[0]):
            session_id = int(result.iloc[index, 0])
            query = result.iloc[index, 1].strip()
            intent = result.iloc[index, 2].strip()
            slot_annotation = result.iloc[index, 3].strip()

            """
            我要听唐朝的 我要听<song>唐朝</singer>的</song>
            来一首k歌之王 来一首<theme>k歌</theme>之</song>王</song>
            伤歌有吗 <emotion>伤歌</toplist>有吗</emotion>
            放一首钢琴的名起来听一下 放一首<style>钢琴</instrument>的名起来听一下</style>
            高阳打电话给高阳 <contact_name>高阳打电话给<contact_name>高阳</contact_name>
            老歌 <theme>老歌</toplist></theme>
            我要听音乐亲子装的歌曲 我要听音乐<song>亲子</emotion>装</song>的歌曲
            请帮我点一首狮子团合唱的的百年孤独 请帮我点一首<singer>狮子团</singer>合唱的的<song>百年</song>孤</song>独
            一把杀猪刀的歌给我 <song>一把<song>杀猪刀</song>的歌给我
            阜新市文化宫阜新市工人文化宫 <destination>阜新市文化宫<destination>阜新市工人文化宫</destination>
            放一个ye ye 放一个<song>ye<song>ye</song>
            龙梅子歌首农民 <singer>龙梅子</singer>歌首<emotion>农民</style>
            只要有你陪在我身边 <song>只要<song>有你陪在我身边</song>
            放一首韩朝舞 放一首<song>韩朝</language>舞</song>
            我想听216车载舞曲 我想听216<song>车载</song>舞</style>曲
            m c天佑的歌曲 m<singer>c<singer>天佑</singer>的歌曲
            我想听大众的我们都一样 我想听<singer>大众</singer>的<song>我们都一</song>样
            导航回家回公司 导航回<custom_destination>家回<custom_destination>公司</custom_destination>
            放旮旯的当你 放<singer>旮[UNK]</singer>的<song>当你</song>
            电话给六四打电话给6454 电话给<contact_name>六四</contact_name>打电话给<phone_num>6454</phone_num>
            dj小浩 <theme>dj小</singer>浩</singer></theme>
            换一首猛歌 换一首<emotion>猛歌</toplist></emotion>         
            """
            # if slot_annotation.count('</') > 1 or slot_annotation.count('<') > 2 or slot_annotation.count('>') == 1:
            #     print(query, slot_annotation)

            if '[UNK]' in slot_annotation:
                sequence = ''
                unk_index = slot_annotation.find('[UNK]')

                for char_index in [-2, -1]:
                    if slot_annotation[unk_index + char_index] in query:
                        sequence += slot_annotation[unk_index + char_index]

                result.iloc[index, 3] = slot_annotation.replace('[UNK]', query[query.find(sequence) + len(sequence)])

            if ' ' in query and slot_annotation.find('<') != -1 and slot_annotation.find('>') != -1:
                space_indexes = []
                for token_index in range(len(query)):
                    if query[token_index] == ' ':
                        space_indexes.append(token_index)

                for space_index in space_indexes:
                    match_sequence = query[space_index - 1] + query[space_index + 1]

                    match_index = slot_annotation.find(match_sequence)
                    if match_index != -1:
                        slot_annotation = slot_annotation[
                                          :match_index + len(match_sequence) - 1] + ' ' + slot_annotation[
                                                                                          match_index + len(
                                                                                              match_sequence) - 1:]

                result.iloc[index, 3] = slot_annotation

        return result

    @staticmethod
    def rule_1(result: pd.DataFrame):
        """
        1、意图设计中未给出的语义槽，不用标注语义槽，但是需要标注意图。
        :param result:
        :return:
        """
        for index in range(result.shape[0]):
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
        contact_name_txt_path = os.path.join(project_dir, 'data', 'input', 'new-slot-dictionaries', 'contact_name.txt')
        destination_txt_path = os.path.join(project_dir, 'data', 'input', 'new-slot-dictionaries', 'destination.txt')

        contact_name_list = []
        with open(contact_name_txt_path, mode='r') as file:
            for line in file.readlines():
                line = line.strip().strip('\n')
                contact_name_list.append(line)

        destination_list = []
        with open(destination_txt_path, mode='r') as file:
            for line in file.readlines():
                line = line.strip().strip('\n')
                destination_list.append(line)

        contact_name_session_id_list = []
        contact_name_query_list = []
        for index in range(result.shape[0]):
            session_id = int(result.iloc[index, 0])
            query = result.iloc[index, 1].strip()

            if query in contact_name_list and query not in destination_list:
                contact_name_query_list.append(query)
                contact_name_session_id_list.append(session_id)

        # print('contact_name_session_id_list_length:{}'.format(len(contact_name_session_id_list)))
        # print(contact_name_session_id_list)
        before_id = None

        phone_call = False
        for index in range(result.shape[0]):
            session_id = int(result.iloc[index, 0])
            query = result.iloc[index, 1].strip()
            intent = result.iloc[index, 2].strip()

            if session_id in contact_name_session_id_list:
                if before_id != session_id:
                    phone_call = False

                if before_id is None or before_id != session_id:
                    before_id = session_id
                    if query in contact_name_query_list:
                        result.iloc[index, 2] = 'OTHERS'
                        result.iloc[index, 3] = query
                    else:
                        if 'phone_call' in intent:
                            phone_call = True
                    continue

                if before_id == session_id:
                    if query in contact_name_query_list:
                        if phone_call:
                            result.iloc[index, 2] = 'phone_call.make_a_phone_call'
                            result.iloc[index, 3] = '<contact_name>' + query + '</contact_name>'
                        else:
                            result.iloc[index, 2] = 'OTHERS'
                            result.iloc[index, 3] = query

                    else:
                        if 'phone_call' in intent:
                            phone_call = True
        return result

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
        8、纠错 针对intent: music.play 同时 slot: singer|song
        :param result:
        :return:
        """
        singer_txt_path = os.path.join(project_dir, 'data', 'input', 'new-slot-dictionaries', 'singer.txt')
        song_txt_path = os.path.join(project_dir, 'data', 'input', 'new-slot-dictionaries', 'song.txt')
        error_pairs_txt_path = os.path.join(project_dir, 'data', 'input', 'new-slot-dictionaries', 'error_pairs.txt')

        singer_list = []
        with open(singer_txt_path, mode='r') as file:
            for line in file.readlines():
                line = line.strip().strip('\n')
                singer_list.append(line)
        singer_list = sorted(singer_list, key=lambda a: len(a), reverse=True)

        song_list = []
        with open(song_txt_path, mode='r') as file:
            for line in file.readlines():
                line = line.strip().strip('\n')
                song_list.append(line)
        song_list = sorted(song_list, key=lambda a: len(a), reverse=True)

        error_pairs_dict = {}
        with open(error_pairs_txt_path, mode='r') as file:
            for line in file.readlines():
                line = line.strip().strip('\n')
                line = line.split('\t')[-1].split('||')
                error_pairs_dict[line[0]] = line[-1]

        for index in range(result.shape[0]):
            session_id = int(result.iloc[index, 0])
            query = result.iloc[index, 1].strip()
            intent = result.iloc[index, 2].strip()
            slot_annotation = result.iloc[index, 3].strip()

            special_sequence = None
            if '的歌曲' in slot_annotation:
                special_sequence = '的歌曲'
            elif '唱的' in slot_annotation:
                special_sequence = '唱的'

            if special_sequence is not None:
                song_sequence = slot_annotation[slot_annotation.find(special_sequence) + len(special_sequence):]
                song_sequence = song_sequence.replace('</song>', '')

                if len(song_sequence) > 1:
                    query_song_sequence = query[query.find(special_sequence) + len(special_sequence):]
                    result.iloc[index, 3] = slot_annotation.replace(song_sequence,
                                                                    '<song>' + query_song_sequence + '</song>')

            if '<singer>' in slot_annotation and '</singer>' in slot_annotation:
                median_sequence = slot_annotation[
                                  slot_annotation.find('<singer>') + len('<singer>'): slot_annotation.find('</singer>')]

                for singer in singer_list:
                    if singer.replace(' ', '') in query.replace(' ', ''):
                        if singer not in slot_annotation:
                            if singer not in query:
                                query_singer = query[query.find(singer[0]): query.find(singer[-1]) + 1]
                                result.iloc[index, 3] = query.replace(query_singer,
                                                                      '<singer>' + query_singer + '||' + singer + '</singer>')
                                break
                            else:
                                result.iloc[index, 3] = query.replace(singer, '<singer>' + singer + '</singer>')
                                break

                        elif '<singer>' + singer + '</singer>' not in slot_annotation and len(singer) > 2:
                            result.iloc[index, 3] = slot_annotation.replace(singer, '<singer>' + singer + '</singer>')
                            break

                if median_sequence not in singer_list:
                    if median_sequence == '张靓玫':
                        result.iloc[index, 3] = slot_annotation.replace('张靓玫', '张靓玫||张靓颖')

                    if median_sequence in error_pairs_dict:
                        result.iloc[index, 3] = slot_annotation.replace(median_sequence,
                                                                        median_sequence + '||' + error_pairs_dict[
                                                                            median_sequence])

            if '<song>' in slot_annotation and '</song>' in slot_annotation:
                median_sequence = slot_annotation[
                                  slot_annotation.find('<song>') + len('<song>'): slot_annotation.find('</song>')]

                best_similarity = Rule.caculate_similarity(token_list=song_list, token=median_sequence)
                if best_similarity and best_similarity not in query and len(median_sequence) >= 3:
                    result.iloc[index, 3] = slot_annotation.replace(median_sequence,
                                                                    median_sequence + '||' + best_similarity)
                    continue

                for song in song_list:
                    if song.replace(' ', '') in query.replace(' ', ''):
                        if song not in slot_annotation:
                            if song not in query:
                                query_song = query[query.find(song[0]): query.find(song[-1]) + 1]
                                result.iloc[index, 3] = query.replace(query_song,
                                                                      '<song>' + query_song + '||' + song + '</song>')
                                break
                            else:
                                result.iloc[index, 3] = query.replace(song, '<song>' + song + '</song>')
                                break

                        elif '<song>' + song + '</song>' not in slot_annotation and len(song) > 2:
                            song_median_sequence = slot_annotation[
                                                   slot_annotation.find('<song>') + len('<song>'):slot_annotation.find(
                                                       '</song>')]
                            if song in song_median_sequence:
                                result.iloc[index, 3] = slot_annotation.replace(song_median_sequence,
                                                                                song_median_sequence + '||' + song)
                                break

                        elif '<song>' + song + '</song>' in slot_annotation:
                            break

                if median_sequence not in song_list:
                    if median_sequence in error_pairs_dict:
                        result.iloc[index, 3] = slot_annotation.replace(median_sequence,
                                                                        median_sequence + '||' + error_pairs_dict[
                                                                            median_sequence])

            if 'trouble is afriend' in query:
                result.iloc[index, 3] = slot_annotation.replace('trouble is afriend',
                                                                '<song>trouble is afriend||trouble is a friend</song>')

        return result

    @staticmethod
    def caculate_similarity(token_list, token):
        """
        计算相似度
        :param token_list:
        :param token:
        :return:
        """
        result = {}
        for token_sample in token_list:
            result[token_sample] = difflib.SequenceMatcher(None, token, token_sample).quick_ratio()

        result = sorted(result.items(), key=lambda a: a[1], reverse=True)

        if token not in token_list and result[0][1] > 0.6 and len(result[0][0]) > 2:
            return result[0][0]
        else:
            return None


if __name__ == '__main__':
    pass
    # ################ 数据探索 ################
    # eda = Eda()
    # eda.match_train_test()

    # ################ 数据处理 ################
    util = Util()

    # 生成新的slot dictionary
    # Slot.generate_new_slot_dictionary(slot_dir=os.path.join(project_dir, 'data', 'input', 'slot-dictionaries'),
    #                                   new_slot_dir=os.path.join(project_dir, 'data', 'input', 'new-slot-dictionaries'))

    # 生成标准的输入数据
    # util.generate_yanxishe_input_data()

    # 生成结果数据
    # util.generate_result()

    # ################ 规则生成 ################

    # 经过规则
    Rule.rule()
