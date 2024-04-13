# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
import torch.distributed as dist
import torch.distributed.nn
import os

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.tanh(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class SimCSEforShortText(nn.Module):
    def __init__(self, pretrained, dropout_prob=0.1, add_batch=True):
        super().__init__()
        conf = BertConfig.from_pretrained(pretrained)
        conf.attention_probs_dropout_prob = dropout_prob
        conf.hidden_dropout_prob = dropout_prob
        self.encoder = BertModel.from_pretrained(pretrained, config=conf)
        self.sim = Similarity(0.05)
        self.mlp = MLP(768, 768)
        self.add_batch = add_batch

    def forward(self, input_ids, token_type_ids, attention_mask, tag_input_ids, tag_token_type_ids, tag_attention_mask,
                mask_indexes):
        tags_output = self.encoder(input_ids=tag_input_ids,
                                   attention_mask=tag_attention_mask,
                                   token_type_ids=tag_token_type_ids).last_hidden_state[:, 0, :]

        sentence_output = self.encoder(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids).last_hidden_state

        z1_list = []
        z2 = tags_output

        for i in range(len(mask_indexes)):
            indexes = mask_indexes[i]
            for index in indexes:
                selected_vectors = [sentence_output[i, idx, :] for idx in index]
                z1_list.append(torch.mean(torch.stack(selected_vectors), dim=0))

        z1 = torch.stack(z1_list, dim=0) if z1_list else None  # z1 是带上下文语境的属性值的池化表征

        z2 = self.mlp(z2)  # z2 是概念词的表征
        if self.add_batch:
            z2 = torch.cat((z2, z1), 0)  # 这里把概念的表征额外的对比学习负样 [2k,768]

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # [k,1,768]  [1,2k,768]   -->  [k,2k]

        if self.add_batch:
            sub_tensor = cos_sim[:, cos_sim.size(0):]
            sub_tensor.diagonal().sub_(1e12)

        loss_fct = nn.CrossEntropyLoss()
        labels = torch.arange(cos_sim.size(0)).long().to(int(os.environ["LOCAL_RANK"]))
        loss = loss_fct(cos_sim, labels)
        loss = torch.mean(loss)

        return loss


class SimCSEforLongText(nn.Module):
    def __init__(self, pretrained, hard_negative_weight=0.1, dropout_prob=0.1, add_batch=True):
        super().__init__()
        conf = BertConfig.from_pretrained(pretrained)
        conf.attention_probs_dropout_prob = dropout_prob
        conf.hidden_dropout_prob = dropout_prob
        self.encoder = BertModel.from_pretrained(pretrained, config=conf)
        self.sim = Similarity(0.05)
        self.add_batch = add_batch
        self.hard_negative_weight = hard_negative_weight

    def forward(self, input_ids, token_type_ids, attention_mask, instruction_input_ids, instruction_token_type_ids,
                instruction_attention_mask):

        per_device_batch_size = input_ids.size(0)  # 下面简写为pdbs
        #         # instruction 前pdbs个是正样本，后pdbs个是困难负例
        instruction_output = self.encoder(input_ids=instruction_input_ids,
                                          attention_mask=instruction_attention_mask,
                                          token_type_ids=instruction_token_type_ids)
        # instruction_output：[2*pdbs,768]
        para_output = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        # para_output：[pdbs,768]

        para_pooler = para_output.pooler_output  # [pbs,768]
        positive_pooler = instruction_output.pooler_output[:per_device_batch_size]  # [pdbs,768]
        negative_pooler = instruction_output.pooler_output[per_device_batch_size:]  # [pdbs,768]

        # 如果使用了分布式训练并且模型处于训练模式
        if dist.is_initialized() and self.training:
            gathered_para_poolers = torch.distributed.nn.all_gather(para_pooler)
            gathered_positive_poolers = torch.distributed.nn.all_gather(positive_pooler)
            gathered_negative_poolers = torch.distributed.nn.all_gather(negative_pooler)
            # torch.distributed.nn.all_gather()的计算返回的结果为 Tutle[torch.tensor], 并且在反向传播的时候这个结果的梯度为求和后的
            para_pooler = torch.cat(gathered_para_poolers, 0)
            positive_pooler = torch.cat(gathered_positive_poolers, 0)
            negative_pooler = torch.cat(gathered_negative_poolers, 0)

        z2 = torch.cat((positive_pooler, negative_pooler), 0)
        # z2: [2*batch_size,768]
        if self.add_batch:
            # 把同一个训练批次内的其余本身也作为负样
            z2 = torch.cat((z2, para_pooler), 0)
            # z2: [3*bs,768]

        cos_sim = self.sim(para_pooler.unsqueeze(1), z2.unsqueeze(0))
        # boardcast : [bs,1,768] , [1,3bs,768]  --> [bs,3bs]  if add_batch
        # boardcast : [bs,1,768] , [1,2bs,768]  --> [bs,2bs]  if not add_batch

        gathered_scale = cos_sim.size(0)
        # 设置困难负例的权重 给每个样本的困难负例的cos_sim +weight
        if self.hard_negative_weight != 0:
            hard_tensor = cos_sim[:, gathered_scale:-gathered_scale]
            hard_tensor.diagonal().add_(self.hard_negative_weight)

        if self.add_batch:
            sub_tensor = cos_sim[:, -gathered_scale:]
            sub_tensor.diagonal().sub_(1e12)

        loss_fct = nn.CrossEntropyLoss()
        labels = torch.arange(gathered_scale).long().to(int(os.environ["LOCAL_RANK"]))
        loss = loss_fct(cos_sim, labels)
        return loss
