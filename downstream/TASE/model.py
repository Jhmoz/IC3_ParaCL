""" Multi span QA model with cl loss"""
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.bert.configuration_bert import BertConfig
import torch.nn.functional as F
import random
import numpy as np

"""
O=0,B=1,I=2
"""

@dataclass
class MultiSpanQuestionAnsweringModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predicted_relations: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertCLForMultiSpanQA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            positive_start_positions: Optional[torch.Tensor] = None,
            positive_end_positions: Optional[torch.Tensor] = None,
            seo_targets: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], MultiSpanQuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )
        total_loss = None
        # last layer hidden states of the previous token: (batch_size, seq_len, hidden_size)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)  # (batch_size, seq_len, 2)

        if positive_start_positions is not None and positive_end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = logits.size(1)
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=ignored_index,
                weight=torch.tensor([1.0, 1.0, 1.0]).to(logits.device),
                reduction='none'
            )
            # Only keep active parts of the loss
            active_loss = token_type_ids.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, seo_targets.view(-1), torch.tensor(loss_fct.ignore_index).type_as(seo_targets)
            )
            loss = loss_fct(active_logits, active_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultiSpanQuestionAnsweringModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


