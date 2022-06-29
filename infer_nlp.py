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
        self.MODEL_PATH = 'microsoft/deberta-v3-large'

        # data
        self.CLASSES_WEIGHTS = []  # weights   # or []
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        self.TOKENIZER.save_pretrained(OUTPUT_DIR + 'tokenizer/')

        self.MAX_LENGTH = 140
        self.BATCH_SIZE = 2
        self.ACCUMULATION_STEPS = 1
        self.N_FOLDS = 5

        # model
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FULL_FINETUNING = True
        self.LR = 1e-5
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

class TestDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].values.tolist()
        self.tokenizer = CONFIG.TOKENIZER
        self.max_length = CONFIG.MAX_LENGTH

    def __len__(self):
        return len(self.texts)

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
        self.dropout = nn.Dropout(p=0.1)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.bert.config.hidden_size*1, 1)

    def loss(self, outputs, targets):
        return nn.BCEWithLogitsLoss(reduction="mean")(outputs, targets.reshape(-1,1))

    def forward(self, input_ids, attention_mask, targets = None):
        hidden = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           return_dict=False)
        hidden = hidden[0]
        attention_mask = attention_mask.unsqueeze(dim=2)
        hidden = hidden * attention_mask
        mean_pooling_embeddings = torch.sum(hidden, 1) / torch.sum(attention_mask, dim=1)
        mean_pooling_embeddings = self.dropout(mean_pooling_embeddings)

        logits1 = self.out(self.dropout1(mean_pooling_embeddings))
        logits2 = self.out(self.dropout2(mean_pooling_embeddings))
        logits3 = self.out(self.dropout3(mean_pooling_embeddings))
        logits4 = self.out(self.dropout4(mean_pooling_embeddings))
        logits5 = self.out(self.dropout5(mean_pooling_embeddings))
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


def infer_fn(model, valid_dataloader):
    val_loss = 0
    model.eval()
    preds = []
    for step, batch in tqdm(enumerate(valid_dataloader),
                            total=len(valid_dataloader),
                            desc='validing'):
        b_input_ids = batch['input_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            y_preds, loss = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
            preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    predictions = np.concatenate(predictions)
    return predictions

valid_ds = TestDataset(test)
valid_dl = DataLoader(valid_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
torch.manual_seed(CONFIG.SEED)
for fold in range(5):
    model = Model()
    model = model.to(device)
    model.load_state_dict(torch.load(f'best_deberta_model_fold_{fold}.pt'))
    valid_probs = infer_fn(model, valid_dl)
    print(valid_probs)