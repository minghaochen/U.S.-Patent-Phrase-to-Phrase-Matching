import warnings
warnings.simplefilter('ignore')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gc
import copy
import time
import random
import re
import numpy as np
import pandas as pd
# pd.set_option('max_columns', None)
# pd.set_option('max_rows', 500)
# pd.set_option('max_colwidth', 200)
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from label_smoothing import LabelSmoothingCrossEntropy

os.makedirs('models', exist_ok=True)
os.makedirs('oofs', exist_ok=True)
os.makedirs('preds', exist_ok=True)

df_train = pd.read_json('nlp_data/train.txt', lines=True)
df_test = pd.read_json('nlp_data/test.txt', lines=True)

print(df_train.shape, df_test.shape)

train_data = list()

for idx, row in tqdm(df_train.iterrows()):
    for entity in row['entity']:
        di = dict()
        di['id'] = f'{row["id"]}_{entity}'
        di['text'] = f'实体: {entity} [SEP] ' + row['content']
        di['entity_len'] = len(entity)
        di['label'] = row['entity'][entity]
        train_data.append(di)

df_train = pd.DataFrame(train_data)
###### 数据示例
#                 id                                               text  label
# 0             1_美国  实体: 美国 [SEP] 3.新疆棉是全球业界公认的高品质天然纤维原料，较好满足了全球范围内...      0
# 1             1_中国  实体: 中国 [SEP] 3.新疆棉是全球业界公认的高品质天然纤维原料，较好满足了全球范围内...      0
# 2             2_德约  实体: 德约 [SEP] 显然，与其指望德约在罗兰-加洛斯击败纳达尔，不如把希望寄托在墨尔本...     -1
# 3          2_梅德韦杰夫  实体: 梅德韦杰夫 [SEP] 显然，与其指望德约在罗兰-加洛斯击败纳达尔，不如把希望寄托在...      1
# 4             2_澳网  实体: 澳网 [SEP] 显然，与其指望德约在罗兰-加洛斯击败纳达尔，不如把希望寄托在墨尔本...      0

test_data = list()

for idx, row in tqdm(df_test.iterrows()):
    for entity in row['entity']:
        di = dict()
        di['id'] = f'{row["id"]}'
        di['text'] = f'实体: {entity} [SEP] ' + row['content']
        di['entity_len'] = len(entity)
        test_data.append(di)

df_test = pd.DataFrame(test_data)

df_train['label'] += 2
df_train.label.value_counts()

# display(df_train['text'].apply(lambda x: len(x)).describe())
# display(df_test['text'].apply(lambda x: len(x)).describe())


class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.SEED = 42
        self.MODEL_PATH = './chinese-roberta-wwm-ext'
        self.NUM_CLASSES = df_train['label'].nunique()

        # data
        self.CLASSES_WEIGHTS = []  # weights   # or []
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        self.MAX_LENGTH = 512
        self.BATCH_SIZE = 4
        self.ACCUMULATION_STEPS = 1
        self.N_FOLDS = 5

        # model
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FULL_FINETUNING = True
        self.LR = 1e-5
        self.N_VALIDATE_DUR_TRAIN = 2
        self.N_WARMUP = 0
        self.SAVE_BEST_ONLY = True
        self.EPOCHS = 2
        self.USE_FGM = False


CONFIG = Config()

# 固定随机性
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
np.random.seed(CONFIG.SEED)
seed_torch(seed=CONFIG.SEED)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = CONFIG.DEVICE


class SentiDataset(Dataset):
    def __init__(self, df, indices, set_type=None):
        super(SentiDataset, self).__init__()

        # df = df.loc[indices]
        self.texts = df['text'].values.tolist()
        self.entity_len = df['entity_len'].values.tolist()
        self.set_type = set_type
        if self.set_type != 'test':
            self.labels = df['label'].values.tolist()

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
        entity_len = torch.zeros_like(attention_mask)
        entity_len[0:4 + self.entity_len[index]] = 1
        content_len = torch.zeros_like(attention_mask)
        content_len[4 + self.entity_len[index]:] = 1

        if self.set_type != 'test':
            return {
                'input_ids': input_ids.long(),
                'attention_mask': attention_mask.long(),
                'entity_len': entity_len,
                'content_len': content_len,
                'labels': torch.tensor(self.labels[index], dtype=torch.long),
            }

        return {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.long(),
            'entity_len': entity_len,
            'content_len': content_len,
        }

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        config = AutoConfig.from_pretrained(CONFIG.MODEL_PATH)
        config.update(
            {
                "hidden_dropout_prob": 0.1,
                "layer_norm_eps": 1e-7,
            }
        )
        self.bert = AutoModel.from_pretrained(CONFIG.MODEL_PATH,  config=config)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.bert.config.hidden_size, CONFIG.NUM_CLASSES)

        # self.multihead_attn = nn.MultiheadAttention(self.bert.config.hidden_size, 8)
        self.multihead_attn = nn.TransformerEncoderLayer(self.bert.config.hidden_size, 8)

    def forward(self, input_ids, attention_mask, entity_len, content_len):
        _, pooled_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     return_dict=False)
        # entity_len = entity_len.unsqueeze(2)
        # entity_emb = _ * entity_len
        # entity_emb = torch.sum(entity_emb, 1) / torch.sum(entity_len,dim=1)
        # content_len = content_len.unsqueeze(2)
        # content_emb = _ * content_len
        # content_emb = torch.sum(content_emb, 1) / torch.sum(content_len, dim=1)
        # last_hidden_states = torch.cat([entity_emb, content_emb, entity_emb-content_emb, torch.mul(entity_emb, content_emb)], 1)
        _ = torch.transpose(_, 0, 1)
        attn_output = self.multihead_attn(_)
        attn_output = torch.transpose(attn_output, 0, 1)

        entity_len = entity_len.unsqueeze(2)
        entity_emb = attn_output * entity_len
        mean_pooling_embeddings = torch.sum(entity_emb, 1) / torch.sum(entity_len, dim=1)


        # last_hidden_states = torch.mean(attn_output, axis=1)
        # output = self.drop(pooled_output)
        # last_hidden_states = torch.mean(_, axis=1)
        return self.out(mean_pooling_embeddings)

def val_fn(model, valid_dataloader, criterion):
    val_loss = 0
    corrects = 0
    model.eval()
    for step, batch in tqdm(enumerate(valid_dataloader),
                            total=len(valid_dataloader),
                            desc='validing'):
        b_input_ids = batch['input_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        b_entity_len = batch['entity_len'].to(device)
        b_content_len = batch['content_len'].to(device)
        with torch.no_grad():
            logits = model(input_ids=b_input_ids, attention_mask=b_attention_mask,
                           entity_len = b_entity_len, content_len = b_content_len)
            loss = criterion(logits, b_labels)
            val_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            corrects += torch.sum(preds == b_labels)
    avg_val_loss = val_loss / len(valid_dataloader)
    avg_val_acc = corrects.cpu().numpy() / len(valid_dataloader) / CONFIG.BATCH_SIZE
    print('Val loss:', avg_val_loss, 'Val acc:', avg_val_acc)
    return avg_val_loss, avg_val_acc

def predict_prob(model, dl):
    probs = []
    model.eval()
    for step, batch in tqdm(enumerate(dl),
                            total=len(dl),
                            desc='infering'):
        b_input_ids = batch['input_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)
        b_entity_len = batch['entity_len'].to(device)
        b_content_len = batch['content_len'].to(device)
        with torch.no_grad():
            logits = model(input_ids=b_input_ids, attention_mask=b_attention_mask,
                           entity_len = b_entity_len, content_len = b_content_len)
            logits = logits.cpu().numpy()
            probs.extend(logits)
    probs = np.array(probs)
    return probs

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='bert.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                # print('fgm attack')
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # print('fgm restore')
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}

def train_fn(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, epoch):
    # we validate config.N_VALIDATE_DUR_TRAIN times during the training loop
    nv = CONFIG.N_VALIDATE_DUR_TRAIN
    temp = len(train_dataloader) // nv
    temp = temp - (temp % 100)
    validate_at_steps = [temp * x for x in range(1, nv + 1)]

    if CONFIG.USE_FGM:
        fgm = FGM(model)

    train_loss = 0
    for step, batch in tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader),
                            desc='training'):
        # print(step)
        # set model.eval() every time during training
        model.train()

        # unpack the batch contents and push them to the device (cuda or cpu).
        b_input_ids = batch['input_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        b_entity_len = batch['entity_len'].to(device)
        b_content_len = batch['content_len'].to(device)

        # forward pass
        logits = model(input_ids=b_input_ids, attention_mask=b_attention_mask,
                       entity_len = b_entity_len, content_len = b_content_len)

        # calculate loss
        loss = criterion(logits, b_labels)
        loss = loss / CONFIG.ACCUMULATION_STEPS
        train_loss += loss.item()

        # backward pass
        loss.backward()

        # fgm attack
        if CONFIG.USE_FGM:
            fgm.attack()
            logits_adv = model(input_ids=b_input_ids, attention_mask=b_attention_mask,
                               entity_len = b_entity_len, content_len = b_content_len)
            loss_adv = criterion(logits_adv, b_labels)
            loss_adv.backward()
            fgm.restore()

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

def metric_fn(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1


folds = StratifiedKFold(n_splits=CONFIG.N_FOLDS, shuffle=True)
for fold, (tr_ind, val_ind) in enumerate(folds.split(df_train, df_train['label'])):

    start_time = time.time()

    if fold != 0:
        continue
    print('fold', fold)

    # train = df_train.loc[tr_ind]
    valid = df_train.loc[val_ind]
    print(df_train.shape)
    print(valid.shape)

    # train_ds = SentiDataset(train, tr_ind)
    train_ds = SentiDataset(df_train, tr_ind)
    valid_ds = SentiDataset(valid, val_ind)
    print(len(train_ds))
    print(len(valid_ds))
    train_dl = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE)
    valid_dl = DataLoader(valid_ds, batch_size=CONFIG.BATCH_SIZE)

    torch.manual_seed(CONFIG.SEED)
    if len(CONFIG.CLASSES_WEIGHTS) > 0:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(CONFIG.CLASSES_WEIGHTS, dtype=torch.float).to(device))
    else:
        # criterion = nn.CrossEntropyLoss()
        print('LabelSmoothingCrossEntropy')
        criterion = LabelSmoothingCrossEntropy(reduction='sum')
    model = Model()
    model = nn.DataParallel(model)
    model = model.to(device)
    # 加载预训练模型
    # model.load_state_dict(torch.load("./pretrained/unsup_epoch_1.pt"))

    if CONFIG.FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(optimizer_parameters, lr=CONFIG.LR)

    # 茶恩
    # 差分学习率
    # module = (
    #     model.module if hasattr(model, "module") else model
    # )
    # no_decay = ["bias", "LayerNorm.weight"]
    # model_param = list(module.named_parameters())
    # bert_param_optimizer = []
    # other_param_optimizer = []
    #
    # for name, para in model_param:
    #     space = name.split('.')
    #     if space[0] == 'bert':
    #         bert_param_optimizer.append((name, para))
    #     else:
    #         other_param_optimizer.append((name, para))
    #
    # optimizer_grouped_parameters = [
    #     # bert other module
    #     {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
    #      "weight_decay": 0.01, 'lr': CONFIG.LR},
    #     {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
    #      "weight_decay": 0.0, 'lr': CONFIG.LR},
    #
    #     # 其他模块，差分学习率
    #     {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
    #      "weight_decay": 0.01, 'lr': CONFIG.LR*100},
    #     {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
    #      "weight_decay": 0.0, 'lr': CONFIG.LR*100},
    # ]
    # optimizer = optim.AdamW(optimizer_grouped_parameters, lr=CONFIG.LR, eps=1e-8)
    # AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=int(opt.warmup_proportion * t_total), num_training_steps=t_total
    # )

    num_training_steps = len(train_dl) * CONFIG.EPOCHS
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=CONFIG.N_WARMUP,
    #     num_training_steps=num_training_steps
    # )
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=CONFIG.N_WARMUP,
                                                num_training_steps=num_training_steps)

    min_avg_val_loss = float('inf')
    for epoch in range(CONFIG.EPOCHS):
        train_fn(model, train_dl, valid_dl, criterion, optimizer, scheduler, epoch)
        avg_val_loss, _ = val_fn(model, valid_dl, criterion)

        if CONFIG.SAVE_BEST_ONLY:
            if avg_val_loss < min_avg_val_loss:
                best_model = copy.deepcopy(model)
                best_val_mse_score = avg_val_loss
                model_name = f'models/fold{fold}_best_model'
                torch.save(best_model.module.state_dict(), model_name + '.pt')
                print(f'--- Best Model. Val loss: {min_avg_val_loss} -> {avg_val_loss}')
                min_avg_val_loss = avg_val_loss

    model = Model()
    model = model.to(device)
    model.load_state_dict(torch.load(f'models/fold{fold}_best_model.pt'))
    valid_probs = predict_prob(model, valid_dl)
    valid_df = valid.copy()
    for i in range(CONFIG.NUM_CLASSES):
        valid_df[f'p{i}'] = valid_probs[:, i]
    valid_df['pred'] = valid_probs.argmax(axis=1)
    valid_df.to_pickle(f'oofs/fold{fold}_oof.pickle')

    acc, f1 = metric_fn(valid['label'], valid_df['pred'])

    used_time = time.time() - start_time

    print(f'fold {fold} score: acc={acc}, f1={f1} used_time: {used_time}')

test_indices = list(range(len(df_test)))
test_data = SentiDataset(df_test, test_indices, set_type='test')
test_dl = DataLoader(test_data, batch_size=CONFIG.BATCH_SIZE )

pred = np.zeros((len(df_test), CONFIG.NUM_CLASSES))

for fold in range(0,1):
    model = Model()
    model = model.to(device)
    model.load_state_dict(torch.load(f'models/fold{fold}_best_model.pt'),False)
    test_probs = predict_prob(model, test_dl)
    pred += test_probs / CONFIG.N_FOLDS
    np.save(f'preds/probs_{fold}', test_probs)

np.save('preds/pred_prob', pred)
df_test['pred'] = np.argmax(pred, axis=1)
# display(df_test.head())

df_test['pred'] -= 2

res = dict()

for idx, row in tqdm(df_test.iterrows()):
    id_ = row['id']
    text = row['text']
    entity = re.sub(r' \[SEP\] .*', '', text).replace('实体: ', '')
    label = row['pred']
    if id_ not in res:
        res[id_] = dict()
    res[id_][entity] = label

tmp = list()
for k in res:
    tmp.append({'id': k, 'result': res[k]})

sub = pd.DataFrame(tmp)
sub.to_csv('section1_head.txt', index=False, sep='\t')

# oof = list()
# for fold in range(CONFIG.N_FOLDS):
#     oof.append(pd.read_pickle(f'oofs/fold{fold}_oof.pickle'))
# df_oof = pd.concat(oof)
# df_oof = df_oof.sort_index()
#
# acc, f1 = metric_fn(df_train['label'], df_oof['pred'])
# print(f'OOF acc={acc}, f1={f1}')