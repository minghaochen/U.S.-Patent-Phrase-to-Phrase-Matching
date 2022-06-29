from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn


class SBERTSingleModel(nn.Module):
    def __init__(self, bert_dir, hidden_size=768, mid_size=512, freeze=False):
        super(SBERTSingleModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = 2

        self.bert = BertModel.from_pretrained(bert_dir)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(mid_size)

        self.linear_a = nn.Linear(hidden_size * 3, mid_size)
        self.classifier_a = nn.Linear(mid_size, 2)

        self.linear_b = nn.Linear(hidden_size * 3, mid_size)
        self.classifier_b = nn.Linear(mid_size, 2)

    def forward(self, source_input_ids, target_input_ids):
        # 0 for [PAD], mask out the padded values
        source_attention_mask = torch.ne(source_input_ids, 0)
        target_attention_mask = torch.ne(target_input_ids, 0)

        # get bert output
        source_embedding = self.bert(source_input_ids, attention_mask=source_attention_mask)
        target_embedding = self.bert(target_input_ids, attention_mask=target_attention_mask)

        # simply take out the [CLS] represention
        # TODO: try different pooling strategies
        source_embedding = source_embedding[1]
        target_embedding = target_embedding[1]

        # concat the source embedding, target embedding and abs embedding as in the original SBERT paper
        abs_embedding = torch.abs(source_embedding - target_embedding)
        context_embedding = torch.cat([source_embedding, target_embedding, abs_embedding], -1)
        context_embedding = self.dropout(context_embedding)

        # get probs for type A
        output_a = self.linear_a(context_embedding)
        output_a = self.bn(output_a)
        output_a = self.relu(output_a)
        output_a = self.dropout(output_a)
        probs_a = self.classifier_a(output_a)

        return probs_a
