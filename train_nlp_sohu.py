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
# from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AdamW
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
        self.BATCH_SIZE = 4
        self.ACCUMULATION_STEPS = 1
        self.N_FOLDS = 5

        # model
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FULL_FINETUNING = True
        self.LR = 1e-7
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
train['context_text'] = train['context_text'].map(lambda s : s.lower())

train['text_a'] = train['anchor'] + '[SEP]' + train['context_text']
train['text_b'] = train['target'] + '[SEP]' + train['context_text']



class TrainDataset(Dataset):
    def __init__(self, df):
        self.texts_a = df['text_a'].values.tolist()
        self.texts_b = df['text_b'].values.tolist()
        self.labels = df['score'].values.tolist()
        self.tokenizer = CONFIG.TOKENIZER
        self.max_length = CONFIG.MAX_LENGTH

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        tokenized_a = self.tokenizer.encode_plus(
            self.texts_a[index],
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        input_ids_a = tokenized_a['input_ids'].squeeze()
        attention_mask_a = tokenized_a['attention_mask'].squeeze()

        tokenized_b = self.tokenizer.encode_plus(
            self.texts_b[index],
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        input_ids_b = tokenized_b['input_ids'].squeeze()
        attention_mask_b = tokenized_b['attention_mask'].squeeze()


        return {
            'input_ids_a': input_ids_a.long(),
            'attention_mask_a': attention_mask_a.long(),
            'input_ids_b': input_ids_b.long(),
            'attention_mask_b': attention_mask_b.long(),
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
                "output_hidden_states": False,
            }
        )
        self.bert = AutoModel.from_pretrained(CONFIG.MODEL_PATH,
                                              config=config)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        mid_size = 512
        self.bn = nn.BatchNorm1d(mid_size)
        self.linear = nn.Linear(self.bert.config.hidden_size * 3, mid_size)
        self.classifier = nn.Linear(mid_size, 1)

    def forward(self, input_ids_a, attention_mask_a,
                input_ids_b, attention_mask_b,
                targets = None):
        hidden_a = self.bert(input_ids=input_ids_a,
                             attention_mask=attention_mask_a,
                             return_dict=False)
        hidden_b = self.bert(input_ids=input_ids_b,
                             attention_mask=attention_mask_b,
                             return_dict=False)

        source_embedding = hidden_a[0][:, 0, :]
        target_embedding = hidden_b[0][:, 0, :]
        abs_embedding = torch.abs(source_embedding - target_embedding)
        context_embedding = torch.cat([source_embedding, target_embedding, abs_embedding], -1)
        context_embedding = self.dropout(context_embedding)

        output = self.linear(context_embedding)
        output = self.bn(output)
        output = self.relu(output)
        output = self.dropout(output)
        logits = self.classifier(output)

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
        batch['input_ids_a'] = batch['input_ids_a'].to(device)
        batch['attention_mask_a'] = batch['attention_mask_a'].to(device)
        batch['input_ids_b'] = batch['input_ids_b'].to(device)
        batch['attention_mask_b'] = batch['attention_mask_b'].to(device)
        batch['labels'] = batch['labels'].to(device)
        with torch.no_grad():
            # forward pass
            y_preds, loss = model(input_ids_a=batch['input_ids_a'], attention_mask_a=batch['attention_mask_a'],
                                 input_ids_b=batch['input_ids_b'], attention_mask_b=batch['attention_mask_b'],
                                 targets=batch['labels'])

            val_loss += loss.item()
            # preds.append(y_preds.to('cpu').numpy())
            preds.append(y_preds.sigmoid().to('cpu').numpy())
    avg_val_loss = val_loss / len(valid_dataloader)
    print('Val loss:', avg_val_loss)
    predictions = np.concatenate(preds)
    predictions = np.concatenate(predictions)
    return avg_val_loss, predictions



def train_fn(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, epoch):
    # we validate config.N_VALIDATE_DUR_TRAIN times during the training loop
    nv = CONFIG.N_VALIDATE_DUR_TRAIN
    temp = len(train_dataloader) // nv
    temp = temp - (temp % 100)
    validate_at_steps = [temp * x for x in range(1, nv + 1)]

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
        batch['input_ids_a'] = batch['input_ids_a'].to(device)
        batch['attention_mask_a'] = batch['attention_mask_a'].to(device)
        batch['input_ids_b'] = batch['input_ids_b'].to(device)
        batch['attention_mask_b'] = batch['attention_mask_b'].to(device)
        batch['labels'] = batch['labels'].to(device)
        # forward pass
        logits, loss = model(input_ids_a=batch['input_ids_a'], attention_mask_a=batch['attention_mask_a'],
                             input_ids_b=batch['input_ids_b'], attention_mask_b=batch['attention_mask_b'],
                             targets=batch['labels'])


        # preds.append(logits.detach().to('cpu').numpy())
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


        # if step in validate_at_steps:
            # print(f'-- Step: {step}')
            # _ = val_fn(model, valid_dataloader, criterion)

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
    train_dl = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=CONFIG.BATCH_SIZE)
    torch.manual_seed(CONFIG.SEED)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    model = Model()
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=CONFIG.LR, weight_decay=1e-3, correct_bias=False)
    num_training_steps = len(train_dl) * CONFIG.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
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
                model_name = f'best_deberta_base_model_fold_{fold}'
                torch.save(model.state_dict(), model_name + '.pt')

    k_fold_score.append(best_score)

print('k_fold_score', k_fold_score)
