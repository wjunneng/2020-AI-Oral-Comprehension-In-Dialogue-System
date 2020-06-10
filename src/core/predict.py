import os
import logging
import argparse
from tqdm import tqdm, trange
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from src.core.utils import init_logger, load_tokenizer, get_intent_labels, get_slot_labels, MODEL_CLASSES

logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_dir,
                                                                  args=args,
                                                                  intent_label_lst=get_intent_labels(args),
                                                                  slot_label_lst=get_slot_labels(args))
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines


def convert_input_file_to_tensor_dataset(lines,
                                         pred_config,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset


def predict_sample(lines: list):
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_dir", default="../../data/yanxishe_model", type=str,
                            help="Path to save, load model")

        parser.add_argument("--batch_size", default=1, type=int, help="Batch size for prediction")
        parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

        pred_config = parser.parse_args()

        # load model and args
        args = get_args(pred_config)
        device = get_device(pred_config)
        model = load_model(pred_config, args, device)
        logger.info(args)

        intent_label_lst = get_intent_labels(args)
        slot_label_lst = get_slot_labels(args)

        # Convert input file to TensorDataset
        pad_token_label_id = args.ignore_index
        tokenizer = load_tokenizer(args)
        dataset = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)

        # Predict
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

        all_slot_label_mask = None
        intent_preds = None
        slot_preds = None

        for batch in data_loader:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "intent_label_ids": None,
                          "slot_labels_ids": None}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]
                outputs = model(**inputs)
                _, (intent_logits, slot_logits) = outputs[:2]

                # Intent Prediction
                if intent_preds is None:
                    intent_preds = intent_logits.detach().cpu().numpy()
                else:
                    intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

                # Slot prediction
                if slot_preds is None:
                    slot_preds = slot_logits.detach().cpu().numpy()
                    all_slot_label_mask = batch[3].detach().cpu().numpy()
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                    all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

        intent_preds = np.argmax(intent_preds, axis=1)

        # 第一大的数
        slot_max_preds = np.array(np.argmax(slot_preds, axis=2)[0])

        slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
        slot_preds_list = [[] for _ in range(slot_max_preds.shape[0])]

        end_token_index = None
        for j in range(slot_max_preds.shape[0]):
            if all_slot_label_mask[0, j] != pad_token_label_id:
                if slot_max_preds[j] not in [1, 2, 3]:
                    end_token_index = slot_max_preds[j]
                slot_preds_list[0].append(slot_label_map[slot_max_preds[j]])

        start_token_index = end_token_index - 1

        slot_second_max_index = np.argmax(slot_preds[:, :, start_token_index], axis=1)[0]
        slot_preds_list[0][slot_second_max_index - 1] = slot_label_map[start_token_index]

        # Write to output file
        for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):
            line = ""
            for word, pred in zip(words, slot_preds):
                if pred == 'O':
                    line = line + word + " "
                else:
                    if pred.startswith('B-'):
                        pred = pred.replace('B-', '<')
                        pred += '>'
                        line = line + "{} {} ".format(pred, word)
                    else:
                        pred = pred.replace('I-', '</')
                        pred += '>'
                        line = line + "{} {} ".format(word, pred)

            return line.strip()

        logger.info("Prediction Done!")

    except:
        return None


def padding_zero_value(intent_label_lst, slot_label_lst, intent_pred, slot_pred, slot_predict):
    """
    填充0值
    :param intent_label_lst: 所有的意图
    :param slot_label_lst: 所有的槽值
    :param intent_pred: [batch_size]    i.e. (32,)
    :param slot_pred: [batch_size, seq_length, slot_size] i.e. (32, 52, 33)
    :param slot_predict: [batch_size, seq_length] i.e. (32, 52)
    :return:
    """
    # intent
    music_play_index = intent_label_lst.index('music.play')
    navigation_navigation_index = intent_label_lst.index('navigation.navigation')
    phone_call_make_a_phone_call_index = intent_label_lst.index('phone_call.make_a_phone_call')

    # slot
    b_music_slot_indexes = [slot_label_lst.index('B-' + i) for i in
                            ['age', 'emotion', 'instrument', 'language', 'scene',
                             'singer', 'song', 'style', 'theme', 'toplist']]
    b_navigation_slot_indexes = [slot_label_lst.index('B-' + i) for i in
                                 ['destination', 'custom_destination', 'origin']]
    b_phone_slot_indexes = [slot_label_lst.index('B-' + i) for i in ['phone_num', 'contact_name']]

    i_music_slot_indexes = [slot_label_lst.index('I-' + i) for i in
                            ['age', 'emotion', 'instrument', 'language', 'scene',
                             'singer', 'song', 'style', 'theme', 'toplist']]
    i_navigation_slot_indexes = [slot_label_lst.index('I-' + i) for i in
                                 ['destination', 'custom_destination', 'origin']]
    i_phone_slot_indexes = [slot_label_lst.index('I-' + i) for i in ['phone_num', 'contact_name']]

    music_slot_indexes = b_music_slot_indexes + i_music_slot_indexes
    navigation_slot_indexes = b_navigation_slot_indexes + i_navigation_slot_indexes
    phone_slot_indexes = b_phone_slot_indexes + i_phone_slot_indexes

    slot_indexes = [i for i in range(len(slot_label_lst))]
    slot_indexes.remove(slot_label_lst.index('PAD'))
    slot_indexes.remove(slot_label_lst.index('UNK'))
    slot_indexes.remove(slot_label_lst.index('O'))

    # TODO：直接填充0的效果不好
    # result = []
    # for intent_sample, slot_sample in zip(intent_pred, slot_pred):
    #     if intent_sample == music_play_index:
    #         delete_list = list(set(slot_indexes) - set(music_slot_indexes))
    #     elif intent_sample == navigation_navigation_index:
    #         delete_list = list(set(slot_indexes) - set(navigation_slot_indexes))
    #     elif intent_sample == phone_call_make_a_phone_call_index:
    #         delete_list = list(set(slot_indexes) - set(phone_slot_indexes))
    #     else:
    #         delete_list = []
    #
    #     for item in delete_list:
    #         slot_sample[:, item] = [i if i > 0 else i for i in slot_sample[:, item]]
    #
    #     result.append(slot_sample)
    #
    # return np.asarray(result)

    result = []
    # TODO: 定点修改
    for intent_pred_sample, slot_pred_sample, slot_predict_sample in zip(intent_pred, slot_pred, slot_predict):
        # intent_pred_sample: (1,)
        # slot_pred_sample: (52, 33)
        # slot_predict_sample: (52,)
        replace = False
        if intent_pred_sample == music_play_index:
            index_list = music_slot_indexes
            re_index_list = list(set(slot_indexes) - set(index_list))
        elif intent_pred_sample == navigation_navigation_index:
            index_list = navigation_slot_indexes
            re_index_list = list(set(slot_indexes) - set(index_list))
        elif intent_pred_sample == phone_call_make_a_phone_call_index:
            index_list = phone_slot_indexes
            re_index_list = list(set(slot_indexes) - set(index_list))
        else:
            result.append(slot_predict_sample)
            continue

        seq_actual_length = 0
        for index in slot_predict_sample[1:]:
            seq_actual_length += 1
            if index == 0:
                break

            if (index not in index_list) and (index not in [slot_label_lst.index('PAD'), slot_label_lst.index('UNK'),
                                                            slot_label_lst.index('O')]):
                replace = True

        if replace:
            actual_slot_pred_sample = slot_pred_sample[1:seq_actual_length, :]
            for index in re_index_list:
                actual_slot_pred_sample[:, index] = -10000

            slot_predict_sample[1: seq_actual_length] = np.argmax(actual_slot_pred_sample, axis=1)

        result.append(slot_predict_sample)

    return result


def predict_batch(pred_config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    logger.info(args)

    intent_label_lst = get_intent_labels(args)
    slot_label_lst = get_slot_labels(args)

    # Convert input file to TensorDataset
    pad_token_label_id = args.ignore_index
    tokenizer = load_tokenizer(args)
    lines = read_input_file(pred_config)
    dataset = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_slot_label_mask = None
    intent_preds = []
    slot_preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "intent_label_ids": None,
                      "slot_labels_ids": None}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

            # Intent Prediction
            intent_pred = np.argmax(intent_logits.detach().cpu().numpy(), axis=1)
            intent_preds.extend(list(intent_pred))

            # Slot prediction
            if slot_preds is None:
                if args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_predict = np.array(model.crf.decode(slot_logits))
                else:
                    slot_predict = slot_logits.detach().cpu().numpy()

                # Padding zero value
                slot_pred = padding_zero_value(intent_label_lst=intent_label_lst, slot_label_lst=slot_label_lst,
                                               intent_pred=intent_pred, slot_pred=slot_logits.detach().cpu().numpy(),
                                               slot_predict=slot_predict)
                # slot_pred = deepcopy(slot_predict)

                slot_preds = deepcopy(slot_pred)
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                if args.use_crf:
                    slot_predict = np.array(model.crf.decode(slot_logits))
                else:
                    slot_predict = np.array(model.crf.decode(slot_logits))

                # Padding zero value
                slot_pred = padding_zero_value(intent_label_lst=intent_label_lst, slot_label_lst=slot_label_lst,
                                               intent_pred=intent_pred, slot_pred=slot_logits.detach().cpu().numpy(),
                                               slot_predict=slot_predict)
                # slot_pred = deepcopy(slot_predict)

                slot_preds = np.append(slot_preds, slot_pred, axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    intent_preds = np.asarray(intent_preds)

    if args.use_crf:
        pass
    else:
        slot_preds = np.argmax(slot_preds, axis=2)

    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    # Write to output file
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):
            line = ""
            for word, pred in zip(words, slot_preds):
                if pred == 'O':
                    line = line + word + " "
                else:
                    if pred.startswith('B-'):
                        pred = pred.replace('B-', '<')
                        pred += '>'
                        line = line + "{} {} ".format(pred, word)
                    else:
                        pred = pred.replace('I-', '</')
                        pred += '>'
                        line = line + "{} {} ".format(word, pred)

            f.write("{}\t{}\n".format(intent_label_lst[intent_pred], line.strip()))

    logger.info("Prediction Done!")


if __name__ == "__main__":
    # ###################### batch ######################
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="../../data/yanxishe/test/sample_pred_in.txt", type=str,
                        help="Input file for prediction")
    parser.add_argument("--output_file", default="../../data/yanxishe/test/sample_pred_out.txt", type=str,
                        help="Output file for prediction")
    parser.add_argument("--model_dir", default="../../data/yanxishe_model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    predict_batch(pred_config)

    # ###################### sample ######################
    # print(predict_sample(lines=[['来', '一', '首', '停', '格']]))
    # print(predict_sample(lines=[['青', '海', '咏', '琳', '的', '歌', '曲']]))
