import collections
import datetime
import json
import random
import itertools
import os
import math
import pickle
import re
import copy
import torch
import argparse
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils import data
from torch.utils.data import RandomSampler
from tqdm.auto import tqdm
from functools import partial
from transformers import (
    RobertaTokenizerFast, BertTokenizerFast, AutoConfig, AutoTokenizer, set_seed
)

from model import BertForMultiSpanQuestionAnswering
from model_roberta import RobertaForMultiSpanQuestionAnswering, DataCollatorForTokenByTokenRelClassification
from transformers import get_scheduler
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup

import utils
from eval import get_entities


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,default="../data/clean")
    parser.add_argument('--model_dir', type=str,default="/home/yingying/models/chinese-roberta-wwm-ext")
    parser.add_argument('--tokenizer_dir', type=str,default=None)
    parser.add_argument('--model_type',type=str,default='roberta')
    parser.add_argument('--decoding_type', type=str,default="JOINT")
    parser.add_argument('--seed', type=int,default=0)
    parser.add_argument('--batch_size', type=int,default=32)
    parser.add_argument('--num_epoch', type=int,default=50)
    parser.add_argument('--learning_rate', type=float,default=1e-5)
    args = parser.parse_args()
    return args


args = get_args()
data_dir = args.data_dir
data_type = os.path.basename(data_dir)
data_files = {
    'train': os.path.join(data_dir, 'train.json'),
    'validation': os.path.join(data_dir, 'valid.json'),
    'test': os.path.join(data_dir, 'test.json')
}
device = "cuda" if torch.cuda.is_available() else "cpu"

# Change "fp16_training" to True to support automatic mixed precision training (fp16)
fp16_training = True
if fp16_training:
    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

# 构造模型
SEED = args.seed
set_seed(SEED)

label_list = ["O", "B", "I"]  # O: 0, B: 1, I: 2  # O = 1, B = 2, I = 4
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}
num_labels = len(label_list)
decoding_type = args.decoding_type  # TAG, JOINT
token_enhance_type = ""  # "none", "adding", ""

# == 构造分词器 ==
tokenizer, model = None, None
model_dir = args.model_dir
if args.tokenizer_dir:
    tokenizer_dir = args.tokenizer_dir
else:
    tokenizer_dir = model_dir

config = AutoConfig.from_pretrained(
    model_dir, num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    finetuning_task='ner'
)
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)
model = BertForMultiSpanQuestionAnswering.from_pretrained(
    model_dir, config=config
).to(device)

# 1. 为分词器加入特殊词
special_tokens_dict = {"additional_special_tokens": ["\001"]}
tokenizer.add_special_tokens(special_tokens_dict)
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# 2. 扩充模型lookup嵌入层词表大小 resize_token_embeddings
model.resize_token_embeddings(len(tokenizer))

print(type(model))
print(type(tokenizer))
# exit(0)
# == 数据处理 ==
# 1. 定义数据集
raw_datasets = load_dataset('json', field='data', data_files=data_files)
train_examples = raw_datasets["train"]
dev_examples = raw_datasets["validation"]
test_examples = raw_datasets["test"]

train_column_names = raw_datasets["train"].column_names
dev_column_names = raw_datasets["validation"].column_names
test_column_names = raw_datasets["test"].column_names

question_column_name = "question"
context_column_name = "context"
label_column_name = "label"
max_seq_length = 512
doc_stride = 128
padding = False
# 2. tokenize: 1）动态pad策略，pad到batch中最长的长度；2）天润查体欧尼truncation策略；3）example to features
train_examples_tokenized = tokenizer(
    train_examples[question_column_name],
    train_examples[context_column_name],
    truncation="only_second",
    max_length=max_seq_length,
    stride=doc_stride,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding=padding,
    is_split_into_words=True,
)
dev_examples_tokenized = tokenizer(
    dev_examples[question_column_name],
    dev_examples[context_column_name],
    truncation="only_second",
    max_length=max_seq_length,
    stride=doc_stride,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding=padding,
    is_split_into_words=True,
)
test_examples_tokenized = tokenizer(
    test_examples[question_column_name],
    test_examples[context_column_name],
    truncation="only_second",
    max_length=max_seq_length,
    stride=doc_stride,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding=padding,
    is_split_into_words=True,
)

def prepare_train_features(examples):
    tokenized_examples = train_examples_tokenized
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    _ = tokenized_examples.pop("offset_mapping")
    tokenized_examples["label_ids"] = []
    tokenized_examples["prev_cur_relations"] = []
    tokenized_examples["token_to_first_token"] = []
    tokenized_examples["example_ids"] = []
    tokenized_examples["feature_ids"] = []
    tokenized_examples["word_ids"] = []
    tokenized_examples["sequence_ids"] = []
    for i, sample_index in enumerate(sample_mapping):
        # i 就是 feature_index
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        try:
            _label = examples[label_column_name][sample_index]
        except Exception as e:
            print(sample_index)
            print(e)
            return
        _word_ids = tokenized_examples.word_ids(i)
        abs_word_ids = []
        first_len = 0
        for _id, _wid in enumerate(_word_ids):
            if sequence_ids[_id] is None:
                abs_word_ids.append(_wid)
                first_len += 1
            elif sequence_ids[_id] == 0:
                first_len += 1
                abs_word_ids.append(_wid)
            else:
                abs_word_ids.append(first_len + _wid)
        # word_id to first token idx using tokenized_examples.token_to_word(i, token_idx)
        word_id2first_token_idx = {}
        for token_idx in range(len(sequence_ids)):
            wid = tokenized_examples.token_to_word(i, token_idx)
            if sequence_ids[token_idx] == 1:
                wid += first_len
            if wid is not None and wid not in word_id2first_token_idx:
                word_id2first_token_idx[wid] = token_idx
        # 搞一个map数组 映射pos 到它所在词的first token的pos
        map_pos = []
        token_to_first_token = []
        for token_idx in range(len(sequence_ids)):
            wid = tokenized_examples.token_to_word(i, token_idx)
            if sequence_ids[token_idx] == 1:
                wid += first_len
            if wid is None:  # 特殊词
                map_pos.append(-100)
                token_to_first_token.append(token_idx)
            else:
                map_pos.append(word_id2first_token_idx[wid])
                token_to_first_token.append(word_id2first_token_idx[wid])
        _previous_word_idx = None
        label_ids = [-100] * token_start_index
        inside_token_indices = [-100] * len(sequence_ids)
        for _i, _word_idx in enumerate(_word_ids[token_start_index:]):
            real_idx = _i + token_start_index
            if _word_idx is None:  # 特殊词
                label_ids.append(-100)
            elif _word_idx != _previous_word_idx:
                label_ids.append(label2id[_label[_word_idx]])
            else:
                label_ids.append(-100)
                if sequence_ids[real_idx] == 1:
                    inside_token_indices[real_idx] = 1
            _previous_word_idx = _word_idx
        tokenized_examples["label_ids"].append(label_ids)  # 0:O/1:B/2:I
        # O = 1, B = 2, I = 4
        # rel 0: (O, O) -> 3
        # rel 1: (O, B) -> 4
        # rel 2: (B, I) (I, I) -> 8 12
        # rel 3: (B, B) (I, B) -> 6 10
        # rel 4: (B, O) (I, O) -> 5 9
        # token内部转移  1.不考虑 设rel=-100;  2. 考虑设rel = 5;
        # question中的token转移和涉及[SEP]特殊词的转移不考虑 设rel=-100
        # label_ids: [1, -100, 2, 2, -100, 0, 1, -100, 2, 2, 0, 0, 0, 0, 0, 0, 0]
        prev_cur_relation = [-100]
        prev_id, raw_rel = -100, None
        # seen_first_tokens = set()
        for idx, label_id in enumerate(label_ids[1:]):
            idx = idx + 1
            # map_pos[idx]为-100时表示idx处的token所处word为特殊词
            cur_id = label_ids[map_pos[idx]] if map_pos[idx] != -100 else -100  # first token id
            if inside_token_indices[idx] == 1:
                prev_cur_relation.append(5)
            elif cur_id == -100:
                prev_cur_relation.append(-100)
            else:
                # label_id: # 0:O/1:B/2:I, O = 1, B = 2, I = 4
                # TODO: 用word中first token的h 代表该word内其他token的h
                #  搞一个map数组 映射pos 到它所在词的first token的pos
                #  e.g. [1, -100, 2, 2, -100, 0, 1, -100, 2, 2, 0, 0, 0, 0, 0, 0, 0]
                #
                raw_rel = int(2 ** prev_id) * 2 + int(2 ** cur_id)
                if raw_rel == 3:
                    prev_cur_relation.append(0)  # 0  'O->O'
                elif raw_rel == 4:
                    prev_cur_relation.append(1)  # 1  'O->B'
                elif raw_rel in [8, 12]:
                    prev_cur_relation.append(2)  # 2  'B->I, I->I'
                elif raw_rel in [6, 10]:
                    prev_cur_relation.append(3)  # 3  'B->B,I->B'
                elif raw_rel in [5, 9]:
                    prev_cur_relation.append(4)  # 4  'B->O,I->O'
                else:
                    print(raw_rel, prev_id, cur_id)
                    prev_cur_relation.append(-100)
            prev_id = cur_id if cur_id != -100 else 0
        tokenized_examples["prev_cur_relations"].append(prev_cur_relation)
        tokenized_examples["token_to_first_token"].append(token_to_first_token)
        tokenized_examples["example_ids"].append(examples["example_id"][sample_index])
        tokenized_examples["feature_ids"].append(i)
        tokenized_examples["word_ids"].append(_word_ids)
        tokenized_examples["sequence_ids"].append(sequence_ids)
    return tokenized_examples


def prepare_eval_features(examples, tokenized_examples):
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    _ = tokenized_examples.pop("offset_mapping")
    tokenized_examples["label_ids"] = []
    tokenized_examples["prev_cur_relations"] = []
    tokenized_examples["token_to_first_token"] = []
    tokenized_examples["example_ids"] = []
    tokenized_examples["feature_ids"] = []
    tokenized_examples["word_ids"] = []
    tokenized_examples["sequence_ids"] = []
    for i, sample_index in enumerate(sample_mapping):
        # i 就是 feature_index
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        try:
            _label = examples[label_column_name][sample_index]
        except Exception as e:
            print(sample_index)
            print(e)
            return
        _word_ids = tokenized_examples.word_ids(i)
        abs_word_ids = []
        first_len = 0
        for _id, _wid in enumerate(_word_ids):
            if sequence_ids[_id] is None:
                abs_word_ids.append(_wid)
                first_len += 1
            elif sequence_ids[_id] == 0:
                first_len += 1
                abs_word_ids.append(_wid)
            else:
                abs_word_ids.append(first_len + _wid)
        # word_id to first token idx using tokenized_examples.token_to_word(i, token_idx)
        word_id2first_token_idx = {}
        for token_idx in range(len(sequence_ids)):
            wid = tokenized_examples.token_to_word(i, token_idx)
            if sequence_ids[token_idx] == 1:
                wid += first_len
            if wid is not None and wid not in word_id2first_token_idx:
                word_id2first_token_idx[wid] = token_idx
        # 搞一个map数组 映射pos 到它所在词的first token的pos
        map_pos = []
        token_to_first_token = []
        for token_idx in range(len(sequence_ids)):
            wid = tokenized_examples.token_to_word(i, token_idx)
            if sequence_ids[token_idx] == 1:
                wid += first_len
            if wid is None:  # 特殊词
                map_pos.append(-100)
                token_to_first_token.append(token_idx)
            else:
                map_pos.append(word_id2first_token_idx[wid])
                token_to_first_token.append(word_id2first_token_idx[wid])
        _previous_word_idx = None
        label_ids = [-100] * token_start_index
        inside_token_indices = [-100] * len(sequence_ids)
        for _i, _word_idx in enumerate(_word_ids[token_start_index:]):
            real_idx = _i + token_start_index
            if _word_idx is None:  # 特殊词
                label_ids.append(-100)
            elif _word_idx != _previous_word_idx:
                label_ids.append(label2id[_label[_word_idx]])
            else:
                label_ids.append(-100)
                if sequence_ids[real_idx] == 1:
                    inside_token_indices[real_idx] = 1
            _previous_word_idx = _word_idx
        tokenized_examples["label_ids"].append(label_ids)  # 0:O/1:B/2:I
        # O = 1, B = 2, I = 4
        # rel 0: (O, O) -> 3
        # rel 1: (O, B) -> 4
        # rel 2: (B, I) (I, I) -> 8 12
        # rel 3: (B, B) (I, B) -> 6 10
        # rel 4: (B, O) (I, O) -> 5 9
        # token内部转移  1.不考虑 设rel=-100;  2. 考虑设rel = 5;
        # question中的token转移和涉及[SEP]特殊词的转移不考虑 设rel=-100
        # label_ids: [1, -100, 2, 2, -100, 0, 1, -100, 2, 2, 0, 0, 0, 0, 0, 0, 0]
        prev_cur_relation = [-100]
        prev_id, raw_rel = -100, None
        # seen_first_tokens = set()
        for idx, label_id in enumerate(label_ids[1:]):
            idx = idx + 1
            # map_pos[idx]为-100时表示idx处的token所处word为特殊词
            cur_id = label_ids[map_pos[idx]] if map_pos[idx] != -100 else -100  # first token id
            if inside_token_indices[idx] == 1:
                prev_cur_relation.append(5)
            elif cur_id == -100:
                prev_cur_relation.append(-100)
            else:
                # label_id: # 0:O/1:B/2:I, O = 1, B = 2, I = 4
                # TODO: 用word中first token的h 代表该word内其他token的h
                #  搞一个map数组 映射pos 到它所在词的first token的pos
                #  e.g. [1, -100, 2, 2, -100, 0, 1, -100, 2, 2, 0, 0, 0, 0, 0, 0, 0]
                #
                raw_rel = int(2 ** prev_id) * 2 + int(2 ** cur_id)
                if raw_rel == 3:
                    prev_cur_relation.append(0)  # 0  'O->O'
                elif raw_rel == 4:
                    prev_cur_relation.append(1)  # 1  'O->B'
                elif raw_rel in [8, 12]:
                    prev_cur_relation.append(2)  # 2  'B->I, I->I'
                elif raw_rel in [6, 10]:
                    prev_cur_relation.append(3)  # 3  'B->B,I->B'
                elif raw_rel in [5, 9]:
                    prev_cur_relation.append(4)  # 4  'B->O,I->O'
                else:
                    print(raw_rel, prev_id, cur_id)
                    prev_cur_relation.append(-100)
            prev_id = cur_id if cur_id != -100 else 0
        tokenized_examples["prev_cur_relations"].append(prev_cur_relation)
        tokenized_examples["token_to_first_token"].append(token_to_first_token)
        tokenized_examples["example_ids"].append(examples["example_id"][sample_index])
        tokenized_examples["feature_ids"].append(i)
        tokenized_examples["word_ids"].append(_word_ids)
        tokenized_examples["sequence_ids"].append(sequence_ids)
    return tokenized_examples

prepare_dev_features=partial(prepare_eval_features,tokenized_examples=dev_examples_tokenized)
prepare_test_features=partial(prepare_eval_features,tokenized_examples=test_examples_tokenized)


# 构造数据集及加载器
train_dataset = train_examples.map(
    prepare_train_features,
    batched=True,
    batch_size=len(train_examples),
    num_proc=None,
    remove_columns=train_column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on train dataset",
)

dev_dataset = dev_examples.map(
    prepare_dev_features,
    batched=True,
    batch_size=len(dev_examples),
    num_proc=None,
    remove_columns=dev_column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on validation dataset",
)
test_dataset = test_examples.map(
    prepare_test_features,
    batched=True,
    batch_size=len(test_examples),
    num_proc=None,
    remove_columns=test_column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on test dataset",
)


def get_train_sampler():
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return RandomSampler(train_dataset)


train_set = train_dataset.remove_columns(['word_ids', 'sequence_ids'])
dev_set = dev_dataset.remove_columns(['word_ids', 'sequence_ids'])
test_set = test_dataset.remove_columns(['word_ids', 'sequence_ids'])
print(len(train_set))
print(len(dev_set))
print(len(test_set))
train_batch_size = args.batch_size
dev_batch_size = 4
test_batch_size = 4

train_sampler = get_train_sampler()
data_collator = DataCollatorForTokenByTokenRelClassification(tokenizer)
train_loader = data.DataLoader(
    train_set, sampler=train_sampler, collate_fn=data_collator,
    batch_size=train_batch_size, pin_memory=True  # shuffle=True,
)
print("len(train_loader):", len(train_loader))
num_steps_per_epoch = len(train_loader)

dev_loader = data.DataLoader(
    dev_set, collate_fn=data_collator,
    batch_size=dev_batch_size, shuffle=False, pin_memory=True
)
test_loader = data.DataLoader(
    test_set, collate_fn=data_collator,
    batch_size=test_batch_size, shuffle=False, pin_memory=True
)

# predict dev_batch labels
rel_to_left_tags = {
    0: {'O'}, 1: {'O'}, 2: {'B', 'I'}, 3: {'B', 'I'}, 4: {'B', 'I'}, 5: {}
}
rel_to_right_tags = {
    0: {'O'}, 1: {'B'}, 2: {'I'}, 3: {'B'}, 4: {'O'}, 5: {}
}
# {'O': 0, 'B': 1, 'I': 2}
rel_to_left_tag_ids = {
    0: [0], 1: [0], 2: [1, 2], 3: [1, 2], 4: [1, 2], 5: {}
}
rel_to_right_tag_ids = {
    0: [0], 1: [1], 2: [2], 3: [1], 4: [0], 5: {}
}


def predict(batch_data, batch_outputs, batch_feature_ids, batch_example_ids):
    results = {}
    batch_size, seq_len = batch_data["input_ids"].shape
    for b in range(batch_size):
        offset = batch_data["token_type_ids"][b].tolist().index(1)
        # if model_type == BERT_BASE:
        #     offset = batch_data["token_type_ids"][b].tolist().index(1)
        # elif model_type == ROBERTA_BASE_SQUAD or model_type == ROBERTA_BASE:
        #     offset = batch_data["input_ids"][b].tolist().index(2) + 2
        # logits: (batch_size, seq_len, 3)
        # _prob, _idx: (seq_len, )
        if decoding_type == "JOINT":
            _tag_prob, _tag_idx = torch.max(batch_outputs.logits[b], dim=1)  # (seq_len, num_label)
            _prev_tag_prob, _prev_tag_idx = torch.concat([batch_outputs.logits.new_zeros(1), _tag_prob[:-1]]), \
                torch.concat([batch_outputs.logits.new_zeros(1).long(), _tag_idx[:-1]])
            #  logits 是 tag prediction的outputs；rel_logits 是transition classifier的outputs
            # rel_logits[b]维度：(seq_len, 6)
            # logits[b]维度：# (seq_len, 3)
            _rel_prob, _rel_idx = torch.max(batch_outputs.rel_logits[b], dim=1)  # (seq_len, 6)
            # _rel_idx: (seq_len, )
            # batch_outputs.rel_logits.new_zeros(seq_len, 6)
            # label2id
            # {'O': 0, 'B': 1, 'I': 2}
            # prev_tag <- valid relations
            # rel_to_left_tags = {
            #     0: {'O'}, 1: {'O'}, 2: {'B', 'I'}, 3: {'B', 'I'}, 4: {'B', 'I'}, 5: {}
            # }
            # rel_to_right_tags = {
            #     0: {'O'}, 1: {'B'}, 2: {'I'}, 3: {'B'}, 4: {'O'}, 5: {}
            # }
            joint_logits = batch_outputs.logits.new_zeros(seq_len, 3)
            joint_logits += batch_outputs.logits[b]
            left_prev_tag0_mask = (_rel_idx.eq(0) + _rel_idx.eq(1)) >= 1
            left_prev_tag1_mask = (_rel_idx.eq(2) + _rel_idx.eq(3) + _rel_idx.eq(4)) >= 1
            left_prev_tag2_mask = (_rel_idx.eq(2) + _rel_idx.eq(3) + _rel_idx.eq(4)) >= 1
            left_rel_mask = left_prev_tag0_mask * _prev_tag_idx.eq(0) + left_prev_tag1_mask * _prev_tag_idx.eq(1) + \
                            left_prev_tag2_mask * _prev_tag_idx.eq(2)

            right_tag0_mask = (_rel_idx.eq(0) + _rel_idx.eq(4)) >= 1
            right_tag1_mask = (_rel_idx.eq(1) + _rel_idx.eq(3)) >= 1
            right_tag2_mask = _rel_idx.eq(2) >= 1

            # joint_logits[:, 0] += 0.5*left_rel_mask*_rel_prob + 0.5*right_tag0_mask*batch_outputs.rel_logits[b][:, 0]
            # joint_logits[:, 1] += 0.5*left_rel_mask*_rel_prob + 0.5*right_tag1_mask*batch_outputs.rel_logits[b][:, 1]
            # joint_logits[:, 2] += 0.5*left_rel_mask*_rel_prob + 0.5*right_tag2_mask*batch_outputs.rel_logits[b][:, 2]

            # joint_logits[:, 0] += 0.5 * left_rel_mask * _rel_prob + 0.5 * right_tag0_mask * _rel_prob
            # joint_logits[:, 1] += 0.5 * left_rel_mask * _rel_prob + 0.5 * right_tag1_mask * _rel_prob
            # joint_logits[:, 2] += 0.5 * left_rel_mask * _rel_prob + 0.5 * right_tag2_mask * _rel_prob

            joint_logits[:, 0] += right_tag0_mask * _rel_prob * left_rel_mask
            joint_logits[:, 1] += right_tag1_mask * _rel_prob * left_rel_mask
            joint_logits[:, 2] += right_tag2_mask * _rel_prob * left_rel_mask

            _prob, _idx = torch.max(joint_logits, dim=1)
        else:
            _prob, _idx = torch.max(batch_outputs.logits[b], dim=1)
        f_index = batch_feature_ids[b].item()
        e_index = batch_example_ids[b].item()
        label = [id2label[idx] for idx in _idx.tolist()]
        results[f_index] = (e_index, label, offset)
    return results


# == 训练 ==
# Training
num_epoch = args.num_epoch
validation = True
test = True
logging_step = 100
# Create an optimizer and learning rate scheduler to fine-tune the model.
# Let’s use the AdamW optimizer from PyTorch
# optimizer
learning_rate = args.learning_rate
# 试一下 别人的optimizer BertAdam？
# Optimizer
# Split weights in two groups, one with weight decay and the other not.
no_decay = ["bias", "LayerNorm.weight"]

num_training_step = num_epoch * len(train_loader)

# schedule
def get_schedulers():
    # _lrs = [1e-4, 1e-5]
    # _lrs = [3e-5]
    _lrs = [learning_rate]
    # _lrs = [1e-4]

    num_training_steps_for_schedulers = [
        # int(num_training_step*0.6), int(num_training_step*0.4)
        num_training_step
    ]
    schedulers, _opts = [], []
    for j, _lr in enumerate(_lrs):
        params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 1e-2,  # 1e-2
                "lr": _lr
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": _lr
            },
        ]
        _opt = AdamW(params)
        _opts.append(_opt)
        schedulers.append(
            get_scheduler(
                name='linear', optimizer=_opt,
                num_warmup_steps=0,
                num_training_steps=num_training_steps_for_schedulers[j]
            )
        )
    last_opt = AdamW(model.parameters(), lr=1e-6)
    _opts.append(last_opt)
    schedulers.append(
        get_scheduler('constant', optimizer=last_opt)
    )
    return schedulers, _opts, num_training_steps_for_schedulers


linear_schedulers, opts, per_schedule_steps = get_schedulers()
if fp16_training:
    opts = [accelerator.prepare_optimizer(opt) for opt in opts]
    model, train_loader = accelerator.prepare(model, train_loader)
model.train()

print("Start Training...")
print(num_training_step)
print(per_schedule_steps)

total_step = 0
cum_step = 0
cur_scheduler = linear_schedulers.pop(0)
cur_optimizer = opts.pop(0)
cur_schedule_steps = per_schedule_steps.pop(0)
cum_step += cur_schedule_steps

dev_f1 = None
best_dev_f1 = -float('inf')

test_f1 = None
best_test_f1 = -float('inf')

model_type = args.model_type

dev_result_dir = f"predictions/dev/{data_type}_{model_type}_{decoding_type}_num_epoch={num_epoch}_batch_size={train_batch_size}_init_lr={learning_rate}_seed={SEED}"
test_result_dir = f"predictions/test/{data_type}_{model_type}_{decoding_type}_num_epoch={num_epoch}_batch_size={train_batch_size}_init_lr={learning_rate}_seed={SEED}"
os.makedirs(dev_result_dir, exist_ok=True)
os.makedirs(test_result_dir, exist_ok=True)

for epoch in range(num_epoch):
    step = 1
    train_loss = train_acc = 0
    for batch in tqdm(train_loader):  # batch_data
        total_step += 1
        if total_step == (cum_step + 1):
            cur_scheduler = linear_schedulers.pop(0)
            cur_optimizer = opts.pop(0)
            cur_schedule_steps = per_schedule_steps.pop(0)
            print("POP")
            cum_step += cur_schedule_steps

        # Load all data into GPU
        batch = {k: i.to(device) for k, i in batch.items()}
        outputs = model(**batch)
        train_loss += outputs.loss
        if fp16_training:
            accelerator.backward(outputs.loss)
        else:
            outputs.loss.backward()

        cur_optimizer.step()
        cur_scheduler.step()
        cur_optimizer.zero_grad()

        # TODO: Apply linear learning rate decay
        # Print training loss and accuracy over past logging step
        if step % logging_step == 0 or step % num_steps_per_epoch == 0:
            # Log metrics inside your training loop to visualize model performance
            loss = train_loss.item() / logging_step
            acc = train_acc / logging_step
            print(
                f"Epoch {epoch + 1} | Step {step} | loss = {loss:.3f}"
            )
            train_loss = 0
            if validation:
                print("Evaluating Dev Set ...")
                model.eval()
                dev_loss = 0
                dev_batch_num = 0
                dev_predicted_results = {}
                with torch.no_grad():
                    for i, dev_batch in enumerate(tqdm(dev_loader)):
                        dev_batch_num += 1
                        dev_batch = {k: i.to(device) for k, i in dev_batch.items()}
                        outputs = model(**dev_batch)
                        # collect predictions
                        feature_ids = dev_batch["feature_ids"]
                        example_ids = dev_batch["example_ids"]
                        # features = [dev_examples_tokenized[feature_index.item()] for feature_index in feature_ids]
                        # examples = [dev_examples[example_index.item()] for example_index in example_ids]
                        predicted_results = predict(dev_batch, outputs, feature_ids, example_ids)
                        dev_predicted_results.update(predicted_results)
                        # outputs.logits: (batch_size, num_label, seq_len)
                        dev_loss += outputs.loss
                example_id2feature_ids = {}
                for k, v in dev_predicted_results.items():
                    example_id2feature_ids.setdefault(v[0], [])
                    example_id2feature_ids[v[0]].append(k)
                for k in example_id2feature_ids:
                    example_id2feature_ids[k].sort()

                dev_predictions = collections.OrderedDict()
                for example_index in range(len(example_id2feature_ids)):
                    previous_word_idx = -1
                    ignored_index = []  # Some example tokens will be disappear after tokenization.
                    valid_labels = []
                    for feature_index in example_id2feature_ids[example_index]:
                        eid, labels, offset = dev_predicted_results[feature_index]
                        assert eid == example_index
                        word_ids = dev_examples_tokenized[feature_index].word_ids
                        word_idx = None
                        for word_idx, label in list(zip(word_ids, labels))[offset:]:
                            if word_idx is None:
                                continue
                            elif word_idx > previous_word_idx:
                                ignored_index += range(previous_word_idx + 1, word_idx)  # ignored word index
                                valid_labels.append(label)
                                while word_idx in ignored_index:
                                    ignored_index.remove(word_idx)
                                previous_word_idx = word_idx
                            else:
                                continue
                        if word_idx is None:  # 原因是context被tokenize之后末尾可能有很多为None的words
                            ignored_index += range(previous_word_idx + 1, len(dev_examples[example_index]["context"]))
                    context = dev_examples[example_index]["context"]
                    ignored_index = sorted(list(set(ignored_index)))
                    for i in ignored_index[::-1]:
                        context = context[:i] + context[i + 1:]

                    # print(len(context), len(valid_labels))
                    assert len(context) == len(valid_labels)

                    predict_entities = get_entities(valid_labels, context)
                    predictions = [x[0] for x in predict_entities]
                    dev_predictions[example_index] = predictions

                golds = utils.read_gold(os.path.join(data_dir, "valid.json"))
                result_data = copy.deepcopy(dev_predictions)
                result = utils.multi_span_evaluate(dev_predictions, golds)

                print(f"Validation | Epoch {epoch + 1} "
                      f"| dev_loss: {dev_loss.item() / dev_batch_num: .3f} "
                      f"| dev_precision: {result['exact_match_precision']:.2f} "
                      f"| dev_recall: {result['exact_match_recall']:.2f} "
                      f"| dev_f1 = {result['exact_match_f1']:.2f}"
                      f"| dev_macro_precision = {result['overlap_precision']:.2f}"
                      f"| dev_macro_recall = {result['overlap_recall']:.2f}"
                      f"| dev_macro_avg_f1 = {result['overlap_f1']:.2f}\n")
                dev_f1 = result['exact_match_f1']
                result_file = os.path.join(dev_result_dir, f'epoch={epoch + 1}_step={step}_dev_f1={dev_f1}')
                print("Dev results is saved in" + result_file)
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    result_data.update({"results": (
                        result['exact_match_precision'], result['exact_match_recall'], result['exact_match_f1'],
                        result['overlap_precision'], result['overlap_recall'], result['overlap_f1'])})
                    best_dev = result
                    json.dump(result_data, open(result_file, "w"))
                elif step % num_steps_per_epoch == 0:
                    result_data.update({"results": (
                        result['exact_match_precision'], result['exact_match_recall'], result['exact_match_f1'],
                        result['overlap_precision'], result['overlap_recall'], result['overlap_f1'])})
                    json.dump(result_data, open(result_file, "w"))

                model.train()

        step += 1


if test:
    print("Evaluating Test Set ...")
    model.eval()
    test_loss = 0
    test_batch_num = 0
    test_predicted_results = {}
    with torch.no_grad():
        for i, test_batch in enumerate(tqdm(test_loader)):
            test_batch_num += 1
            test_batch = {k: i.to(device) for k, i in test_batch.items()}
            outputs = model(**test_batch)
            # collect predictions
            feature_ids = test_batch["feature_ids"]
            example_ids = test_batch["example_ids"]
            predicted_results = predict(test_batch, outputs, feature_ids, example_ids)
            test_predicted_results.update(predicted_results)
            # outputs.logits: (batch_size, num_label, seq_len)
            test_loss += outputs.loss
    example_id2feature_ids = {}
    for k, v in test_predicted_results.items():
        example_id2feature_ids.setdefault(v[0], [])
        example_id2feature_ids[v[0]].append(k)
    for k in example_id2feature_ids:
        example_id2feature_ids[k].sort()

    test_predictions = collections.OrderedDict()
    for example_index in range(len(example_id2feature_ids)):
        previous_word_idx = -1
        ignored_index = []  # Some example tokens will be disappear after tokenization.
        test_labels = []
        for feature_index in example_id2feature_ids[example_index]:
            eid, labels, offset = test_predicted_results[feature_index]
            assert eid == example_index
            word_ids = test_examples_tokenized[feature_index].word_ids
            word_idx = None
            for word_idx, label in list(zip(word_ids, labels))[offset:]:
                if word_idx is None:
                    continue
                elif word_idx > previous_word_idx:
                    ignored_index += range(previous_word_idx + 1, word_idx)  # ignored word index
                    test_labels.append(label)
                    while word_idx in ignored_index:
                        ignored_index.remove(word_idx)
                    previous_word_idx = word_idx
                else:
                    continue
            if word_idx is None:  # 原因是context被tokenize之后末尾可能有很多为None的words
                ignored_index += range(previous_word_idx + 1, len(test_examples[example_index]["context"]))
        context = test_examples[example_index]["context"]
        ignored_index = sorted(list(set(ignored_index)))
        for i in ignored_index[::-1]:
            context = context[:i] + context[i + 1:]

        assert len(context) == len(test_labels)

        predict_entities = get_entities(test_labels, context)
        predictions = [x[0] for x in predict_entities]
        test_predictions[example_index] = predictions

    golds = utils.read_gold(os.path.join(data_dir, "test.json"))
    result_data = copy.deepcopy(test_predictions)
    result = utils.multi_span_evaluate(test_predictions, golds)

    print(f"Tese | Epoch {epoch + 1} "
          f"| test_loss: {test_loss.item() / test_batch_num: .3f} "
          f"| test_precision: {result['exact_match_precision']:.2f} "
          f"| test_recall: {result['exact_match_recall']:.2f} "
          f"| test_f1 = {result['exact_match_f1']:.2f}"
          f"| test_macro_precision = {result['overlap_precision']:.2f}"
          f"| test_macro_recall = {result['overlap_recall']:.2f}"
          f"| test_macro_avg_f1 = {result['overlap_f1']:.2f}\n")
    test_f1 = result['exact_match_f1']
    result_file = os.path.join(test_result_dir, f'epoch={epoch + 1}_step={step}_test_f1={test_f1}')
    print("Test results is saved in" + result_file)
    if test_f1 > best_test_f1:
        best_test_f1 = test_f1
        result_data.update({"results": (
            result['exact_match_precision'], result['exact_match_recall'], result['exact_match_f1'],
            result['overlap_precision'], result['overlap_recall'], result['overlap_f1'])})
        best_test = result
        json.dump(result_data, open(result_file, "w"))
    elif step % num_steps_per_epoch == 0:
        result_data.update({"results": (
            result['exact_match_precision'], result['exact_match_recall'], result['exact_match_f1'],
            result['overlap_precision'], result['overlap_recall'], result['overlap_f1'])})
        json.dump(result_data, open(result_file, "w"))

print("Saving Model ...")
model_output_dir = f"{model_type}_{decoding_type}_num_epoch={num_epoch}_batch_size={train_batch_size}_init_lr={learning_rate}_{data_type}"
os.makedirs("saved_models", exist_ok=True)
model_save_dir = "saved_models/{}".format(model_output_dir)
model.save_pretrained(model_save_dir)
print("Saving Metrics ...")
metrics_output_dir = os.path.join("metrics", model_output_dir)
os.makedirs(metrics_output_dir,exist_ok=True)
with open("metrics_test.json","w",encoding="utf-8") as f:
    json.dump(best_test, f, indent=4)
with open("metrics_dev.json","w",encoding="utf-8") as f:
    json.dump(best_dev, f, indent=4)
