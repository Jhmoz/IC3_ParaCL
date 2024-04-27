""" Multi span QA model"""
import pickle
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import ModelOutput, PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
# from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
from transformers import (
    PreTrainedTokenizerBase, RobertaForQuestionAnswering
)


# Data collator
@dataclass
class DataCollatorForTokenByTokenRelClassification(DataCollatorMixin):

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        # features is a list, each feature is a dict with info provided by dataset's __getitem__.
        # info: (input_ids, token_type_ids, attention_mask, label_ids, prev_cur_relations)
        # features consist of batch size of items
        # print(type(features))
        # print(type(features[0]))
        # print(features[0].keys())
        # exit(0)
        import torch
        token_by_token_rel_name = "prev_cur_relations"
        token_map_name = "token_to_first_token"
        label_name = "label_ids"
        labels = [feature["label_ids"] for feature in features] if "label_ids" in features[0].keys() else None
        relations = [feature[token_by_token_rel_name] for feature in features] \
            if token_by_token_rel_name in features[0].keys() else None
        first_tokens = [feature[token_map_name] for feature in features] \
            if token_map_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch
        # print([feature["example_ids"] for feature in features])
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
            batch[token_by_token_rel_name] = [
                list(rel) + [self.label_pad_token_id] * (sequence_length - len(rel)) for rel in relations
            ]
            batch[token_map_name] = [
                list(ft) + [self.label_pad_token_id] * (sequence_length - len(ft)) for ft in first_tokens
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]
            batch[token_by_token_rel_name] = [
                [self.label_pad_token_id] * (sequence_length - len(rel)) + list(rel) for rel in relations
            ]
            batch[token_map_name] = [
                [self.label_pad_token_id] * (sequence_length - len(ft)) + list(ft) for ft in first_tokens
            ]
        # for k, v in batch.items():
        #     print(k)
        #     batch[k] = torch.tensor(v, dtype=torch.int64)

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


@dataclass
class MultiSpanQuestionAnsweringModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    rel_logits: torch.FloatTensor = None
    sequence_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def append_1_length_zeros(tensor):
    B, _, D = tensor.shape
    zero_embed = tensor.new_zeros((B, 1, D))
    fixed_tensor = torch.cat((tensor, zero_embed), 1)
    return fixed_tensor


class RobertaForMultiSpanQuestionAnswering(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        print("******** config.num_labels:", config.num_labels)
        self.num_labels = config.num_labels  # 3, 1: start, 2: end, 0: neither
        self.rel_num_labels = 6  # 0: (0, 0), 1: (0, 1), 2: (1, 0), 3:(1, 1), 4: (2, 0), 5: inside token rel
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # self.first_token_outputs = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.prev_cur_rel_outputs = nn.Linear(2 * config.hidden_size, self.rel_num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # O = 1, B = 2, I = 4
        # (O, O) -> 3
        # (O, B) -> 4
        # (B, I) (I, I) -> 8 12
        # (B, B) (I, B) -> 6 10
        # (B, O) (I, O) -> 5 9
        # token内部转移  1.不考虑 设rel=-100;  2. 考虑设rel = 5;
        # question中的token转移和涉及[SEP]特殊词的转移不考虑 设rel=-100
        # label_ids: [1, -100, 2, 2, -100, 0, 1, -100, 2, 2, 0, 0, 0, 0, 0, 0, 0]
        self.relations = nn.ModuleList()
        self.relations.append(nn.Linear(2 * config.hidden_size, config.hidden_size))  # 0  'O->O'
        self.relations.append(nn.Linear(2 * config.hidden_size, config.hidden_size))  # 1  'O->B'
        self.relations.append(nn.Linear(2 * config.hidden_size, config.hidden_size))  # 2  'B->I,I->I'
        self.relations.append(nn.Linear(2 * config.hidden_size, config.hidden_size))  # 3  'B->B,I->B'
        self.relations.append(nn.Linear(2 * config.hidden_size, config.hidden_size))  # 4  'B->O,I->O'
        self.relations.append(nn.Linear(2 * config.hidden_size, config.hidden_size))
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            example_ids: Optional[torch.Tensor] = None,
            feature_ids: Optional[torch.Tensor] = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            label_ids: Optional[torch.Tensor] = None,
            prev_cur_relations: Optional[torch.Tensor] = None,
            token_to_first_token: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultiSpanQuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        # sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        # sequence_output = self.dropout(sequence_output)
        # logits = self.qa_outputs(sequence_output)  # (batch_size, seq_len, 3)
        total_loss = None
        # """
        bz, seq_len, hz = outputs[0].shape
        # 按照 token_to_first_token 变换 outputs[0] 也即encoder的outputs
        # 用 token_to_first_token 做 map_orders
        # (B, seq_len)
        appended = append_1_length_zeros(outputs[0])
        map_mask = token_to_first_token.eq(-100)  # 搞掉pad词
        map_orders = token_to_first_token.masked_fill(map_mask, seq_len)
        first_token_sequence_output = torch.cat([
            appended[i].index_select(0, map_orders[i]).unsqueeze(0) for i in range(bz)
        ])
        mapped_sequence_output = outputs[0]
        # mapped_sequence_output = outputs[0] + first_token_sequence_output

        # first_token_enhanced = torch.cat((outputs[0], first_token_sequence_output), dim=2)
        # mapped_sequence_output = self.first_token_outputs(first_token_enhanced)
        # mapped_sequence_output = self.dropout(mapped_sequence_output)

        # map_mask = token_to_first_token.eq(-100)  # 搞掉pad词
        # map_orders = token_to_first_token.masked_fill(map_mask, seq_len)
        # print("map_orders:", map_orders)
        # appended = append_1_length_zeros(outputs[0])
        # mapped_sequence_output = torch.cat([
        #     appended[i].index_select(0, map_orders[i]).unsqueeze(0) for i in range(bz)
        # ])
        zero_embed = outputs[0].new_zeros(bz, 1, hz)
        # appended_sequence_output = append_1_length_zeros(outputs[0])  # (B, seq_len+1, D)
        appended_sequence_output = append_1_length_zeros(mapped_sequence_output)
        # prev_sequence_output = torch.cat((zero_embed, outputs[0]), dim=1)  # (B, seq_len+1, D)
        prev_sequence_output = torch.cat((zero_embed, mapped_sequence_output), dim=1)  # (B, seq_len+1, D)

        # 用于prev_token和cur_token关系分类的outputs
        prev_outputs = prev_sequence_output[:, :-1, :]
        cur_outputs = outputs[0]
        prev_cur_outputs = torch.cat((prev_outputs, cur_outputs), dim=2)
        rel_logits = self.prev_cur_rel_outputs(prev_cur_outputs)  # (batch_size, seq_len, 5)
        ignore_index = -100
        if prev_cur_relations is not None:
            rel_loss_fct = nn.CrossEntropyLoss(
                ignore_index=ignore_index,
                reduction='none'
            )
            rel_loss = rel_loss_fct(rel_logits.reshape(-1, self.rel_num_labels), prev_cur_relations.reshape(-1)).mean()
            total_loss = rel_loss

        # 预测 prev_cur_relations (batch_size, seq_len)
        _prob, _idx = torch.max(rel_logits, dim=2)  # _idx shape: (batch_size, seq_len)
        # (bz, seq_len)
        rel_masks = prev_cur_relations.eq(ignore_index)
        predicted_relations = _idx.masked_fill(rel_masks, ignore_index)
        sequence_output = outputs[0].new_zeros(outputs[0].shape)
        irrel_mask = predicted_relations.ne(-1).long()
        for rel_idx in range(self.rel_num_labels):
            _mask = predicted_relations.ne(rel_idx)
            irrel_mask -= (~_mask).long()
            # (B, seq_len)
            cur_orders = torch.arange(seq_len).long().repeat(bz, 1).to(outputs[0].device).masked_fill(_mask, seq_len)
            rel_masked_sequence_output = torch.cat([
                appended_sequence_output[i].index_select(0, cur_orders[i]).unsqueeze(0) for i in range(bz)
            ])
            prev_orders = torch.arange(seq_len).long().repeat(bz, 1).to(outputs[0].device).masked_fill(_mask, 0)
            rel_masked_prev_sequence_output = torch.cat([
                prev_sequence_output[i].index_select(0, prev_orders[i]).unsqueeze(0) for i in range(bz)
            ])
            relation_layer = self.relations[rel_idx]
            _H = relation_layer(torch.cat((rel_masked_prev_sequence_output, rel_masked_sequence_output), dim=2))
            sequence_output += _H

        x_orders = torch.arange(seq_len).long().repeat(bz, 1).to(outputs[0].device) \
            .masked_fill(~(irrel_mask.bool()), seq_len)  # 把已经整过rel的位置都设为seq_len
        sx = torch.cat([appended_sequence_output[i].index_select(0, x_orders[i]).unsqueeze(0) for i in range(bz)], 0)
        # print("sx shape:", sx.shape)  # (bz, seq_len, hidden_size)
        sequence_output += sx
        logits = self.qa_outputs(sequence_output)  # (batch_size, seq_len, 3)
        # """
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep activate parts of the loss
            active_loss = attention_mask.view(-1) == 1  # (batch_size * num_labels, )  num_labels = 3
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, label_ids.view(-1), torch.tensor(loss_fct.ignore_index).type_as(label_ids)
            )
            # loss1:
            # (batch_size, ) -> (1,)
            class_loss = loss_fct(active_logits, active_labels)
            # total_loss = class_loss
            if total_loss is None:
                total_loss = class_loss
            else:
                total_loss += class_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MultiSpanQuestionAnsweringModelOutput(
            loss=total_loss,
            logits=logits,
            rel_logits=rel_logits,
            sequence_output=outputs[0],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
