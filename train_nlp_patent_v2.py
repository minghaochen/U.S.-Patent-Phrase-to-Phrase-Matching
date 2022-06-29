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

INPUT_DIR = '/kaggle/input/us-patent-phrase-to-phrase-matching/'
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.DEBUG = False
        self.model = 1
        if self.model == 1:
            self.SEED = 42
            self.MODEL_TYPE = 'PatentSBERTa'
            self.MODEL_PATH = 'AI-Growth-Lab/PatentSBERTa'
        elif self.model == 2:
            self.SEED = 1994
            self.MODEL_TYPE = 'deberta_v3'
            self.MODEL_PATH = 'microsoft/deberta-v3-large'
        else:
            self.SEED = 2022
            self.MODEL_TYPE = 'deberta-v2'
            self.MODEL_PATH = 'microsoft/deberta-v2-xlarge'
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        self.MAX_LENGTH = 140
        self.BATCH_SIZE = 8
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.LR = 1e-5
        self.EPOCHS=5

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
seed_everything(seed=CONFIG.SEED)


# ====================================================
# Data Loading
# ====================================================
if CONFIG.DEBUG:
    train = pd.read_csv('train.csv', nrows=1000)
else:
    train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')
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
train['context_text'] = train['context_text'].map(lambda s : s.lower())
test['context_text'] = test['context_text'].map(lambda s : s.lower())

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

        return {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.long(),
            'labels': torch.tensor(self.labels[index], dtype=torch.float),
        }


class Model(nn.Module):
    def __init__(self, training_mode = 'MSE'):
        super(Model, self).__init__()
        config = AutoConfig.from_pretrained(CONFIG.MODEL_PATH)
        config.update(
            {
                "hidden_dropout_prob": 0.1,
                "layer_norm_eps": 1e-7,
                # "output_hidden_states": True,
            }
        )
        self.bert = AutoModel.from_pretrained(CONFIG.MODEL_PATH,
                                              config=config)
        # self.dropout = nn.Dropout(p=0.1)
        # self.dropout1 = nn.Dropout(p=0.1)
        # self.dropout2 = nn.Dropout(p=0.2)
        # self.dropout3 = nn.Dropout(p=0.3)
        # self.dropout4 = nn.Dropout(p=0.4)
        # self.dropout5 = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.bert.config.hidden_size*1, 1)

        if training_mode == 'MSE':
            self.loss = nn.MSELoss(reduction="mean")
        else:
            self.loss = nn.BCEWithLogitsLoss(reduction="mean")


    def forward(self, input_ids, attention_mask, targets = None):
        hidden = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           return_dict=False)

        mean_pooling_embeddings = hidden[1]
        mean_pooling_embeddings = self.dropout(mean_pooling_embeddings)

        logits1 = self.out(self.dropout1(mean_pooling_embeddings)).reshape(-1,)
        logits2 = self.out(self.dropout2(mean_pooling_embeddings)).reshape(-1,)
        logits3 = self.out(self.dropout3(mean_pooling_embeddings)).reshape(-1,)
        logits4 = self.out(self.dropout4(mean_pooling_embeddings)).reshape(-1,)
        logits5 = self.out(self.dropout5(mean_pooling_embeddings)).reshape(-1,)
        logits = (logits1+logits2+logits3+logits4+logits5)/5
        loss = 0
        if targets is not None:
            loss1 = self.loss(logits1, targets)
            loss2 = self.loss(logits2, targets)
            loss3 = self.loss(logits3, targets)
            loss4 = self.loss(logits4, targets)
            loss5 = self.loss(logits5, targets)
            loss = (loss1+loss2+loss3+loss4+loss5)/5

            return logits, loss

        return logits, loss


def val_fn(model, valid_dataloader, mode = 'MSE'):
    val_loss = 0
    model.eval()
    preds = []
    for step, batch in tqdm(enumerate(valid_dataloader),
                            total=len(valid_dataloader),
                            desc='validing'):
        b_input_ids = batch['input_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        with torch.no_grad():
            y_preds, loss = model(input_ids=b_input_ids, attention_mask=b_attention_mask, targets=b_labels)
            val_loss += loss.item()
            if mode == 'MSE':
                preds.append(y_preds.to('cpu').numpy())
            else:
                preds.append(y_preds.sigmoid().to('cpu').numpy())
    avg_val_loss = val_loss / len(valid_dataloader)
    print('Val loss:', avg_val_loss)
    predictions = np.concatenate(preds)
    return avg_val_loss, predictions

def train_fn(model, train_dataloader, optimizer, scheduler, mode = 'MSE'):

    train_loss = 0
    preds = []
    train_labels =[]
    model.train()
    for step, batch in tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader),
                            desc='training', ncols = 80):
        train_labels.append(batch['labels'].detach().numpy())

        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        batch['labels'] = batch['labels'].to(device)

        # forward pass
        logits, loss = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], targets=batch['labels'])

        if mode == 'MSE':
            preds.append(logits.detach().to('cpu').numpy())
        else:
            preds.append(logits.sigmoid().detach().to('cpu').numpy())

        train_loss += loss.item()
        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    avg_train_loss = train_loss / len(train_dataloader)
    print('Training loss:', avg_train_loss)

    predictions = np.concatenate(preds)
    train_labels = np.concatenate(train_labels)
    score = get_score(train_labels, predictions)
    print('\n training score', score)


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
train['score_map'] = train['score'].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})
encoder = LabelEncoder()
train['anchor_map'] = encoder.fit_transform(train['anchor'])
kf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=CONFIG.SEED)
for n, (_, valid_index) in enumerate(kf.split(train, train['score_map'], groups=train['anchor_map'])):
    train.loc[valid_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)

for loss_fuc in ['MSE']:
    k_fold_score = []
    OOF = train[['fold','score']].copy()
    OOF['pred'] = 0
    for fold in range(4):

        train_ds = TrainDataset(train[train['fold'] != fold])
        valid_ds = TrainDataset(train[train['fold'] == fold])
        print(train[train['fold'] != fold].score.mean(), train[train['fold'] == fold].score.mean())
        valid_labels = train[train['fold'] == fold].score.values
        print(len(train_ds))
        print(len(valid_ds))
        train_dl = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE)
        valid_dl = DataLoader(valid_ds, batch_size=CONFIG.BATCH_SIZE)
        torch.manual_seed(CONFIG.SEED)
        model = Model(training_mode=loss_fuc)
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
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=1000,
                                                    num_training_steps=num_training_steps)

        best_score = 0
        for epoch in range(CONFIG.EPOCHS):

            train_fn(model, train_dl, optimizer, scheduler)
            avg_val_loss, predictions = val_fn(model, valid_dl)
            score = get_score(valid_labels, predictions)

            if best_score < score:
                best_score = score
                model_name = f'{CONFIG.MODEL_TYPE}_{loss_fuc}_fold_{fold}'
                torch.save(model.state_dict(), model_name + '.pt')
                OOF.loc[OOF['fold']==fold,'pred'] = predictions

            print('\n best score', best_score)

        k_fold_score.append(best_score)

    print('k_fold_score', k_fold_score)
    OOF.to_csv(f'OOF_MODEL_{CONFIG.MODEL_TYPE}_{loss_fuc}.csv', index=None)
    score = get_score(OOF['score'], OOF['pred'])
    print('oof score', score)


