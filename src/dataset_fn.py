import json
import torch
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
from typing import List, Dict


def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    for i in range(len(content)):
        content[i] = content[i].replace('\n', '')
    return content


class TemplateSummaryDataset(Dataset):
    def __init__(self, infor_path):
        self.para_path = infor_path
        self.information = read_json(infor_path)

    def __len__(self):
        return len(self.information)

    def __getitem__(self, index):
        information = self.information[index]
        sentence = information['sentence']
        tags = information['cl_tags']
        mask_indexes = information['mask_indexes']
        sample = {
            'sentence': sentence,
            'mask_indexes': mask_indexes,
            'tag': tags
        }
        return sample


def collate_fn_tag(batch: List[Dict[str, any]], tokenizer) -> Dict[str, torch.Tensor]:
    sentences = [example['sentence'] for example in batch]
    tags = [example['tag'] for example in batch]
    mask_indexes = [example['mask_indexes'] for example in batch]
    tokens_sentences = tokenizer(sentences, truncation=True, padding=True, return_tensors='pt', max_length=512)
    cl_tags = []
    for tag in tags:
        cl_tags = cl_tags + tag
    tokens_tags = tokenizer(cl_tags, truncation=True, padding=True, return_tensors='pt')
    tokens_sentences['tag_input_ids'] = tokens_tags['input_ids']
    tokens_sentences['tag_token_type_ids'] = tokens_tags['token_type_ids']
    tokens_sentences['tag_attention_mask'] = tokens_tags['attention_mask']
    tokens_sentences['mask_indexes'] = mask_indexes
    return tokens_sentences


class InstructionParaDataset(Dataset):
    def __init__(self, para_path):
        self.para_path = para_path
        self.pairs = read_json(para_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        sample = {
            'paragraph': pair['paragraphs'],
            'positive': pair['instructions'],
            'negative': pair['negative']
        }
        return sample


def collate_fn_para(batch: List[Dict[str, any]], tokenizer) -> Dict[str, torch.Tensor]:
    paragraphs = [example['paragraph'] for example in batch]
    positives = [example['positive'] for example in batch]
    negatives = [example['negative'] for example in batch]
    instructions = positives + negatives
    tokenized_paragraphs = tokenizer(paragraphs, truncation=True, padding=True, return_tensors='pt', max_length=512)
    tokenized_instructions = tokenizer(instructions, truncation=True, padding="longest", return_tensors='pt',
                                       max_length=512)
    tokenized_paragraphs['instruction_input_ids'] = tokenized_instructions['input_ids']
    tokenized_paragraphs['instruction_token_type_ids'] = tokenized_instructions['token_type_ids']
    tokenized_paragraphs['instruction_attention_mask'] = tokenized_instructions['attention_mask']
    return tokenized_paragraphs
