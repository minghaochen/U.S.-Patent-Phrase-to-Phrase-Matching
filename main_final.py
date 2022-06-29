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
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIR = '/kaggle/input/us-patent-phrase-to-phrase-matching/'
INPUT_DIR = ''
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# deberta_v3_small microsoft/deberta-v3-small done
# bert_for_patents  anferico/bert-for-patents done
# microsoft/deberta-v3-base done
# microsoft/deberta-v3-large
# microsoft/deberta-base done
# microsoft/deberta-large
# microsoft/deberta-v3-xsmall done
# microsoft/deberta-xlarge
# roberta-base done
# xlm-roberta-base done
# roberta-large
# xlm-roberta-large
# nghuyong/ernie-2.0-en done

class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.DEBUG = False
        self.SEED = 42
        self.MODEL_TYPE = 'deberta_v3_small'
        self.MODEL_PATH = 'microsoft/deberta-v3-small'
        self.BATCH_SIZE = 8
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.LR = 1e-5
        self.N_WARMUP = 0
        self.EPOCHS = 5


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
    train = pd.read_csv(INPUT_DIR + 'train.csv', nrows=1000)
else:
    train = pd.read_csv(INPUT_DIR + 'train.csv')
test = pd.read_csv(INPUT_DIR + 'test.csv')
submission = pd.read_csv(INPUT_DIR + 'sample_submission.csv')
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
torch.save(cpc_texts, OUTPUT_DIR + "cpc_texts.pth")
train['context_text'] = train['context'].map(cpc_texts)
test['context_text'] = test['context'].map(cpc_texts)
train['context_text'] = train['context_text'].map(lambda s: s.lower())
test['context_text'] = test['context_text'].map(lambda s: s.lower())

train['text'] = 'anchor:' + train['anchor'] + '[SEP]' + 'target:' + train['target'] + '[SEP]' + 'context:' + train[
    'context_text']
test['text'] = 'anchor:' + test['anchor'] + '[SEP]' + 'target:' + test['target'] + '[SEP]' + 'context:' + test[
    'context_text']

# ====================================================
# tokenizer
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(CONFIG.MODEL_PATH)
tokenizer.save_pretrained(OUTPUT_DIR + 'tokenizer/')
CONFIG.TOKENIZER = tokenizer

# ====================================================
# Define max_len
# ====================================================
lengths_dict = {}

lengths = []
tk0 = tqdm(cpc_texts.values(), total=len(cpc_texts))
for text in tk0:
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    lengths.append(length)
lengths_dict['context_text'] = lengths

for text_col in ['anchor', 'target']:
    lengths = []
    tk0 = tqdm(train[text_col].fillna("").values, total=len(train))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    lengths_dict[text_col] = lengths

CONFIG.MAX_LENGTH = max(lengths_dict['anchor']) + max(lengths_dict['target']) \
                    + max(lengths_dict['context_text']) + 4  # CLS + SEP + SEP + SEP
LOGGER.info(f"max_len: {CONFIG.MAX_LENGTH}")


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
    def __init__(self, training_mode='MSE'):
        super(Model, self).__init__()

        self.bert = AutoModelForSequenceClassification.from_pretrained(CONFIG.MODEL_PATH, num_labels=1)

        if training_mode == 'MSE':
            self.loss = nn.MSELoss(reduction="mean")
        else:
            self.loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, input_ids, attention_mask, targets=None):
        hidden = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask)
        loss = 0
        if targets is not None:
            loss = self.loss(hidden.logits.reshape(-1, 1), targets.reshape(-1, 1))
            return hidden.logits, loss

        return hidden.logits, loss


def val_fn(model, valid_dataloader, mode='MSE'):
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
                preds.append(y_preds.reshape(-1, ).to('cpu').numpy())
            else:
                preds.append(y_preds.reshape(-1, ).sigmoid().to('cpu').numpy())
    avg_val_loss = val_loss / len(valid_dataloader)
    print('Val loss:', avg_val_loss)
    predictions = np.concatenate(preds)
    return avg_val_loss, predictions


def train_fn(model, train_dataloader, optimizer, scheduler, mode='MSE'):
    train_loss = 0
    preds = []
    train_labels = []
    model.train()
    for step, batch in tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader),
                            desc='training', ncols=80):
        train_labels.append(batch['labels'].detach().numpy())

        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        batch['labels'] = batch['labels'].to(device)

        logits, loss = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                             targets=batch['labels'])

        preds.append(logits.reshape(-1, ).detach().to('cpu').numpy())

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

for loss_fuc in ['MSE', 'BCE']:
    k_fold_score = []
    OOF = train[['fold', 'score']].copy()
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

        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.LR,
                                      betas=(0.9, 0.999),
                                      weight_decay=1e-3)
        num_training_steps = len(train_dl) * CONFIG.EPOCHS
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_training_steps // 10,
                                                    num_training_steps=num_training_steps)

        best_score = 0
        for epoch in range(CONFIG.EPOCHS):

            train_fn(model, train_dl, optimizer, scheduler, loss_fuc)
            avg_val_loss, predictions = val_fn(model, valid_dl, loss_fuc)
            score = get_score(valid_labels, predictions)

            if best_score < score:
                best_score = score
                model_name = f'autodl-tmp/{CONFIG.MODEL_TYPE}_{loss_fuc}_fold_{fold}'
                torch.save(model.state_dict(), model_name + '.pt')
                OOF.loc[OOF['fold'] == fold, 'pred'] = predictions

            print('\n best score', best_score)

        k_fold_score.append(best_score)
        del model, train_dl, valid_dl, train_ds, valid_ds
        torch.cuda.empty_cache()
        gc.collect()

    print('k_fold_score', k_fold_score)
    OOF.to_csv(f'OOF_MODEL_{CONFIG.MODEL_TYPE}_{loss_fuc}.csv', index=None)
    score = get_score(OOF['score'], OOF['pred'])
    print('oof score', score)


