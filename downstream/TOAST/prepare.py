import json
from tqdm import trange
import os
import pickle
import random
from transformers import BertTokenizerFast

# == 构造分词器 ==
model_dir = "/home/captain/models/bert-base-chinese/"
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
# 1. 为分词器加入特殊词
special_tokens_dict = {"additional_special_tokens": ["\001"]}
tokenizer.add_special_tokens(special_tokens_dict)


def token_encoding_to_start_end_pair(word_id, token_encoding):
    if token_encoding.tokens[word_id] == "[UNK]":
        start, end = token_encoding.offsets[word_id]
        return start, end
    else:
        token_start_id = token_encoding.word_ids.index(word_id)
        token_end_id = len(token_encoding.word_ids) - token_encoding.word_ids[::-1].index(word_id) - 1
        start = token_encoding.offsets[token_start_id][0]
        end = token_encoding.offsets[token_end_id][1]
        return start, end


def word_ids_to_start_end_pairs(word_ids, token_encoding, orig_text):
    res = []
    last_word_id = -1
    for word_id in word_ids:
        if word_id == last_word_id:
            continue
        res.append(token_encoding_to_start_end_pair(word_id, token_encoding))
        last_word_id = word_id
    return res


def extract_words(tokenized_text, text):
    n = len(text)
    words = []
    start_end_pairs = word_ids_to_start_end_pairs(tokenized_text.word_ids, tokenized_text, text)
    # 处理None char
    fixed_pairs = []
    i = 0
    while i < n:
        if len(start_end_pairs) > 0:
            start, end = start_end_pairs.pop(0)
            for j in range(i, start):
                fixed_pairs.append((j, j+1))
            fixed_pairs.append((start, end))
            i = end
        else:
            for j in range(i, n):
                fixed_pairs.append((j, j+1))
            i = n
    for s, e in fixed_pairs:
        words.append(text[s: e])
    return words, fixed_pairs
    # start, end = 0, 0
    # last_word_id = -1
    # for i in range(n):
    #     word_id = tokenized_text.char_to_word(i)
    #     if word_id is not None:
    #         if word_id == last_word_id:
    #             end = i + 1
    #         else:
    #             words.append((start, end))
    #             start = i
    #     else:
    #
    #     last_word_id = word_id


def process(input_file, out_dir, splits=[0.9, 0.05, 0.05], version="0.1"):
    # input_file = "../data/clean-v0.1/input_data_v3.pkl"
    result = {'version': version, 'data': []}
    orig_data = pickle.load(open(input_file, 'rb'))
    n = len(orig_data['querys'])
    tokenized_querys = tokenizer(orig_data['querys'], add_special_tokens=False)
    tokenized_contexts = tokenizer(orig_data['contexts'], add_special_tokens=False)
    for i in trange(n):
        example_id = i
        _id = str(example_id)
        #_type = ""
        _type = orig_data["goldtype"][i]
        question, _ = extract_words(tokenized_querys[i], orig_data['querys'][i])
        # breakpoint()
        context, pairs = extract_words(tokenized_contexts[i], orig_data['contexts'][i])
        # 找到各answer_span都hit到哪些words了
        label = ['O'] * len(context)
        answer_starts = orig_data['pos_answer_starts'][i]
        answer_ends = orig_data['pos_answer_ends'][i]
        for start, end in zip(answer_starts, answer_ends):
            hits = []
            hitting = False
            for j, (ps, pe) in enumerate(pairs):
                if not hitting:
                    if ps <= start < pe:  # hits
                        hits.append(j)
                        hitting = True
                        if end <= pe:
                            hitting = False
                else:
                    if end >= pe:
                        hits.append(j)
                    else:
                        hitting = False

            label[hits[0]] = 'B'
            for j in hits[1:]:
                label[j] = 'I'

        result['data'].append(
            {
                'id': _id, 'type': _type, 'example_id': example_id,
                'question': list(question), 'context': list(context), 'label': label
            }
        )

    random.shuffle(result['data'])
    
    clean_all = {'version': version, 'data': result['data']}
    train = {'version': version, 'data': result['data'][:int(splits[0] * n)]}
    valid = {'version': version, 'data': result['data'][int(splits[0] * n): int(sum(splits[:2]) * n)]}
    test = {'version': version, 'data': result['data'][int(sum(splits[:2]) * n):]}

    for i in range(len(train['data'])):
        train['data'][i]['id'] = str(i)
        train['data'][i]['example_id'] = i

    for i in range(len(valid['data'])):
        valid['data'][i]['id'] = str(i)
        valid['data'][i]['example_id'] = i

    for i in range(len(test['data'])):
        test['data'][i]['id'] = str(i)
        test['data'][i]['example_id'] = i

    with open(os.path.join(out_dir, "all.json"), "w") as outf:
        json.dump(clean_all, outf)
    with open(os.path.join(out_dir, "train.json"), "w") as outf:
        json.dump(train, outf)

    with open(os.path.join(out_dir, "valid.json"), "w") as outf:
        json.dump(valid, outf)

    with open(os.path.join(out_dir, "test.json"), "w") as outf:
        json.dump(test, outf)


process("./data/input_data_v5_type.pkl", "./data/clean_type")
