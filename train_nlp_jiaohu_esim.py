# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import shutil
import string
import pickle
import random
import joblib
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import torch
print(f"torch.__version__: {torch.__version__}")
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIR = './'
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.DEBUG = False
        self.SEED = 42
        self.MODEL_PATH = 'microsoft/deberta-v3-base'

        # data
        self.CLASSES_WEIGHTS = []  # weights   # or []
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        self.MAX_LENGTH = 140
        self.BATCH_SIZE = 8
        self.ACCUMULATION_STEPS = 1
        self.N_FOLDS = 5

        # model
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FULL_FINETUNING = True
        self.LR = 5e-6
        self.N_VALIDATE_DUR_TRAIN = 2
        self.N_WARMUP = 0
        self.SAVE_BEST_ONLY = True
        self.EPOCHS = 5
        self.USE_FGM = False

CONFIG = Config()

# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score


def get_logger(filename=OUTPUT_DIR + 'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=42)


# ====================================================
# Data Loading
# ====================================================
if CONFIG.DEBUG:
    train = pd.read_csv(INPUT_DIR+'train.csv', nrows=1000)
else:
    train = pd.read_csv(INPUT_DIR + 'train.csv')
test = pd.read_csv(INPUT_DIR+'test.csv')
submission = pd.read_csv(INPUT_DIR+'sample_submission.csv')
print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")
print(f"submission.shape: {submission.shape}")

# ====================================================
# CPC Data
# ====================================================
def get_cpc_texts():
    contexts = []
    pattern = '[A-Z]\d+'
    for file_name in os.listdir('./cpc-data/CPCSchemeXML202105'):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']:
        with open(f'./cpc-data/CPCTitleList202202/cpc-section-{cpc}_20220201.txt', encoding='utf-8') as f:
            s = f.read()
        pattern = f'{cpc}\t\t.+'
        result = re.findall(pattern, s)
        cpc_result = result[0].lstrip(pattern)
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f'{context}\t\t.+'
            result = re.findall(pattern, s)
            results[context] = cpc_result + ". " + result[0].lstrip(pattern)
    return results


cpc_texts = get_cpc_texts()
torch.save(cpc_texts, OUTPUT_DIR+"cpc_texts.pth")
train['context_text'] = train['context'].map(cpc_texts)
test['context_text'] = test['context'].map(cpc_texts)


train['text'] = 'anchor:' + train['anchor'] + '[SEP]' + 'target:' + train['target'] + '[SEP]' + 'context:' + train['context_text']
test['text'] = 'anchor:' + test['anchor'] + '[SEP]' + 'target:' + test['target'] + '[SEP]' + 'context:' + test['context_text']



class TrainDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].values.tolist()
        self.labels = df['score'].values.tolist()
        self.tokenizer = CONFIG.TOKENIZER
        self.max_length = CONFIG.MAX_LENGTH

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        tokenized = self.tokenizer.encode_plus(
            self.texts[index],
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids'].squeeze()
        attention_mask = tokenized['attention_mask'].squeeze()

        sep_index = np.argwhere(input_ids.numpy() == 2)
        anchor_len = torch.zeros_like(attention_mask)
        anchor_len[1:sep_index[0][0]] = 1
        target_len = torch.zeros_like(attention_mask)
        target_len[sep_index[0][0] + 1:sep_index[1][0]] = 1
        context_len = torch.zeros_like(attention_mask)
        context_len[sep_index[1][0] + 1:sep_index[2][0]] = 1

        return {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.long(),
            'anchor_len': anchor_len,
            'target_len': target_len,
            'context_len': context_len,
            'labels': torch.tensor(self.labels[index], dtype=torch.float),
        }



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        config = AutoConfig.from_pretrained(CONFIG.MODEL_PATH)
        config.update(
            {
                "hidden_dropout_prob": 0.1,
                "layer_norm_eps": 1e-7,
                "output_hidden_states": True,
            }
        )
        self.bert = AutoModel.from_pretrained(CONFIG.MODEL_PATH,
                                              config=config)
        self.multihead_attn = nn.TransformerEncoderLayer(self.bert.config.hidden_size, 8)

        self._projection = nn.Sequential(nn.Linear(4*self.bert.config.hidden_size, self.bert.config.hidden_size),
                                         nn.ReLU())

        self._classification = nn.Sequential(nn.Dropout(0.1),
                                             nn.Linear(4*self.bert.config.hidden_size, self.bert.config.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(0.1),
                                             nn.Linear(self.bert.config.hidden_size, 1))


    def forward(self, input_ids, attention_mask,
                anchor_len, target_len,context_len,
                targets = None):
        hidden = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           return_dict=False)
        hidden = hidden[0]

        hidden = torch.transpose(hidden, 0, 1)
        hidden = self.multihead_attn(hidden)
        hidden = torch.transpose(hidden, 0, 1)

        anchor_len = anchor_len.unsqueeze(2)
        anchor_emb = hidden * anchor_len
        # anchor_emb = nn.functional.normalize(anchor_emb, dim=-1)
        target_len = target_len.unsqueeze(2)
        target_emb = hidden * target_len
        # target_emb = nn.functional.normalize(target_emb, dim=-1)

        similarity_matrix_0 = anchor_emb.bmm(target_emb.transpose(2, 1).contiguous())
        similarity_matrix_1 = target_emb.bmm(anchor_emb.transpose(2, 1).contiguous())

        # attn_mask_0 = similarity_matrix_0 == 0
        # attn_mask_1 = similarity_matrix_1 == 0
        # similarity_matrix_0.masked_fill_(attn_mask_0, -1e9)
        # similarity_matrix_1.masked_fill_(attn_mask_1, -1e9)
        #
        # similarity_matrix_0 = torch.exp(similarity_matrix_0)
        # similarity_matrix_1 = torch.exp(similarity_matrix_1)


        similarity_matrix_0 = similarity_matrix_0 / (similarity_matrix_0.sum(dim=-1, keepdim=True) + 1e-13)
        similarity_matrix_1 = similarity_matrix_1 / (similarity_matrix_1.sum(dim=-1, keepdim=True) + 1e-13)

        attended_anchor_emb = similarity_matrix_0.bmm(target_emb)
        attended_target_emb = similarity_matrix_1.bmm(anchor_emb)

        enhanced_anchor_emb = torch.cat([anchor_emb,
                                         attended_anchor_emb,
                                         anchor_emb - attended_anchor_emb,
                                         anchor_emb * attended_anchor_emb],
                                         dim=-1)
        enhanced_target_emb = torch.cat([target_emb,
                                         attended_target_emb,
                                         target_emb - attended_target_emb,
                                         target_emb * attended_target_emb],
                                         dim=-1)

        projected_anchor = self._projection(enhanced_anchor_emb)
        projected_target = self._projection(enhanced_target_emb)

        # projected_anchor = torch.transpose(projected_anchor, 0, 1)
        # projected_anchor = self.multihead_attn(projected_anchor)
        # projected_anchor = torch.transpose(projected_anchor, 0, 1)
        #
        # projected_target = torch.transpose(projected_target, 0, 1)
        # projected_target = self.multihead_attn(projected_target)
        # projected_target = torch.transpose(projected_target, 0, 1)

        anchor_emb = projected_anchor * anchor_len
        anchor_emb_avg = torch.sum(anchor_emb, 1) / torch.sum(anchor_len,dim=1)
        target_emb = projected_target * target_len
        target_emb_avg = torch.sum(target_emb, 1) / torch.sum(target_len, dim=1)

        anchor_emb_max, _ = anchor_emb.max(dim=1)
        target_emb_max, _ = target_emb.max(dim=1)

        v = torch.cat([anchor_emb_avg, target_emb_avg, anchor_emb_max, target_emb_max], dim=1)

        logits = self._classification(v)

        loss = 0
        if targets is not None:
            # loss = nn.MSELoss()(logits, targets)
            loss = nn.BCEWithLogitsLoss(reduction="mean")(logits, targets.reshape(-1,1))
            return logits, loss
        return logits, loss





def val_fn(model, valid_dataloader, criterion):
    val_loss = 0
    model.eval()
    preds = []
    for step, batch in tqdm(enumerate(valid_dataloader),
                            total=len(valid_dataloader),
                            desc='validing'):
        b_input_ids = batch['input_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)
        batch['anchor_len'] = batch['anchor_len'].to(device)
        batch['target_len'] = batch['target_len'].to(device)
        batch['context_len'] = batch['context_len'].to(device)
        b_labels = batch['labels'].to(device)
        with torch.no_grad():
            y_preds, loss = model(input_ids=b_input_ids, attention_mask=b_attention_mask,
                                  anchor_len=batch['anchor_len'], target_len=batch['target_len'],
                                  context_len = batch['context_len'],
                                  targets=b_labels)
            val_loss += loss.item()
            # preds.append(y_preds.to('cpu').numpy())
            preds.append(y_preds.sigmoid().to('cpu').numpy())
    avg_val_loss = val_loss / len(valid_dataloader)
    print('Val loss:', avg_val_loss)
    predictions = np.concatenate(preds)
    # predictions = np.concatenate(predictions)
    return avg_val_loss, predictions


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

def train_fn(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, epoch):
    # we validate config.N_VALIDATE_DUR_TRAIN times during the training loop
    nv = CONFIG.N_VALIDATE_DUR_TRAIN
    temp = len(train_dataloader) // nv
    temp = temp - (temp % 100)
    validate_at_steps = [temp * x for x in range(1, nv + 1)]

    if CONFIG.USE_FGM:
        fgm = FGM(model)

    train_loss = 0
    preds = []
    train_labels =[]
    for step, batch in tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader),
                            desc='training', ncols = 80):
        train_labels.append(batch['labels'].detach().numpy())
        # set model.eval() every time during training
        model.train()

        # unpack the batch contents and push them to the device (cuda or cpu).
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        batch['anchor_len'] = batch['anchor_len'].to(device)
        batch['target_len'] = batch['target_len'].to(device)
        batch['context_len'] = batch['context_len'].to(device)
        batch['labels'] = batch['labels'].to(device)
        # forward pass
        logits, loss = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                             anchor_len = batch['anchor_len'], target_len = batch['target_len'],
                             context_len=batch['context_len'],
                             targets=batch['labels'])

        preds.append(logits.sigmoid().detach().to('cpu').numpy())
        train_loss += loss.item()

        # backward pass
        loss.backward()

        if (step + 1) % CONFIG.ACCUMULATION_STEPS == 0:
            # update weights
            optimizer.step()
            # clear accumulated gradients
            optimizer.zero_grad()
            # update scheduler
            scheduler.step()


    avg_train_loss = train_loss / len(train_dataloader)
    print('Training loss:', avg_train_loss)

    predictions = np.concatenate(preds)
    # predictions = np.concatenate(predictions)
    train_labels = np.concatenate(train_labels)
    score = get_score(train_labels, predictions)
    print('\n training score', score)

anchors = train.anchor.unique()
np.random.shuffle(anchors)
val_prop = 0.2
val_sz = int(len(anchors)*val_prop)
val_anchors_0 = anchors[:val_sz]
val_anchors_1 = anchors[val_sz:2*val_sz]
val_anchors_2 = anchors[2*val_sz:3*val_sz]
val_anchors_3 = anchors[3*val_sz:4*val_sz]
val_anchors_4 = anchors[4*val_sz:]
val_anchors = [val_anchors_0,val_anchors_1,val_anchors_2,val_anchors_3,val_anchors_4]

k_fold_score = []
for fold in range(5):
    is_val = np.isin(train.anchor, val_anchors[fold])
    idxs = np.arange(len(train))
    val_idxs = idxs[is_val]
    trn_idxs = idxs[~is_val]
    print(train.iloc[trn_idxs].score.mean(), train.iloc[val_idxs].score.mean())

    train_ds = TrainDataset(train.iloc[trn_idxs])
    valid_ds = TrainDataset(train.iloc[val_idxs])
    valid_labels = train['score'].iloc[val_idxs].values
    print(len(train_ds))
    print(len(valid_ds))
    train_dl = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE)
    valid_dl = DataLoader(valid_ds, batch_size=CONFIG.BATCH_SIZE)
    torch.manual_seed(CONFIG.SEED)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    model = Model()
    model = model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=CONFIG.LR)

    num_training_steps = len(train_dl) * CONFIG.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=CONFIG.N_WARMUP,
                                                num_training_steps=num_training_steps)

    min_avg_val_loss = float('inf')
    best_score = 0
    for epoch in range(CONFIG.EPOCHS):
        print('\n best score', best_score)

        train_fn(model, train_dl, valid_dl, criterion, optimizer, scheduler, epoch)
        avg_val_loss, predictions = val_fn(model, valid_dl, criterion)
        # scoring
        score = get_score(valid_labels, predictions)
        print('\n val score', score)

        if CONFIG.SAVE_BEST_ONLY:
            if best_score < score:
                best_score = score
                model_name = f'best_deberta_ESIM_BCE_model_fold_{fold}'
                torch.save(model.state_dict(), model_name + '.pt')

    k_fold_score.append(best_score)

print('k_fold_score', k_fold_score)
