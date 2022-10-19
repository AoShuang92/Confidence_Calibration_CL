from pickle import TRUE
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
import os
import random
import copy
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.optim.lr_scheduler import _LRScheduler
import csv
import tensorflow_probability as tfp
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_everything(seed=12):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class WarmUpLR(_LRScheduler):
    """
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def write_csv(filename, data):
    with open(filename, 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

class SentimentClassifier(torch.nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
    def forward(self, input_ids, attention_mask,token_type_ids):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
            )
        
        output = self.drop(pooled_output[1])
        return self.out(output)


class CELossWith_MHCLS(torch.nn.Module):
    
    def __init__(self, classes=1000, smoothing=0.1, ignore_index=-1, confls=None, confls_weight=None):
        super(CELossWith_MHCLS, self).__init__()
        self.smoothing = smoothing if confls is None else smoothing + confls/confls_weight
        self.complement = 1.0 - smoothing
        self.cls = classes
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.ignore_index = ignore_index

    def forward(self, logits, target, ):
        with torch.no_grad():
            oh_labels = F.one_hot(target.to(torch.int64), num_classes = self.cls).contiguous()
            smoothen_ohlabel = oh_labels * self.complement  + self.smoothing / self.cls
            smoothen_ohlabel = (smoothen_ohlabel/smoothen_ohlabel.sum(1)[:,None]) # to normalise the distr into sum 1
        
        logs = self.log_softmax(logits[target!=self.ignore_index])
        return -torch.sum(logs * smoothen_ohlabel[target!=self.ignore_index], dim=1).mean()


def get_threshold_confidence(model_rank, trainloader_rank, init_ratio=0.80):  
    conf_list_all = []
    correct = 0
    total = 0
    with torch.no_grad():
        for d in trainloader_rank:
            inputs = d["input_ids"].squeeze(1).to(device)
            attention_mask = d["mask"].squeeze(1).to(device)
            token_type_ids =d['token_type_ids'].squeeze(1).to(device)
            targets = d["targets"].to(device)
            outputs = model_rank(input_ids=inputs,attention_mask=attention_mask,token_type_ids = token_type_ids)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            conf = F.softmax(outputs, dim=1)
            conf_list = [conf_each[int(targets[i])] for i, conf_each in enumerate(conf)]
            conf_list_all.extend(torch.tensor(conf_list))

        print('ranked model acc:',correct / total)
    conf_list_all = torch.tensor(conf_list_all)
    sample_total = len(conf_list_all)
    for mu in np.arange(1, 0, -0.001):
        sample_mu = len(conf_list_all[conf_list_all>mu])
        ratio = sample_mu / sample_total
        if ratio >= init_ratio:
            break

    return mu, conf_list_all

def get_threshold_scaler_ad_sample(ad, init_ratio=0.80):     
    ad_std_list = []
    for i in range (len(ad)):
        ad_std = np.std(ad[i])
        ad_std_list.append(ad_std)
    ad_std_list = np.array(ad_std_list)
    std_norm_list = []
    
    for k in range (len(ad_std_list)):
        norm_std = (ad_std_list[k] - min(ad_std_list)) / (max(ad_std_list)-min(ad_std_list))
        std_norm_list.append(norm_std)

    std_norm_list = np.array(std_norm_list)
    
    conf_list_all = torch.tensor(std_norm_list)
    
    sample_total = len(conf_list_all)
    for mu in np.arange(1, 0, -0.001):
        sample_mu = len(conf_list_all[conf_list_all>mu])
        ratio = sample_mu / sample_total
        if ratio >= init_ratio:
            break

    return mu, conf_list_all


def train(model, trainloader, criterion, optimizer, epoch, scheduler, args):
    model.train()
    for d in trainloader:
        inputs = d["input_ids"].squeeze(1).to(device)
        attention_mask = d["mask"].squeeze(1).to(device)
        token_type_ids =d['token_type_ids'].squeeze(1).to(device)
        targets = d["targets"].to(device)
        # print("target",targets[:10] )
        conf_list = d['conf'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=inputs,attention_mask=attention_mask,token_type_ids = token_type_ids)
        loss = criterion(outputs[conf_list>=args.mu], targets[conf_list>=args.mu])
        # print("loss", loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()

        
        
def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    logits_list = []
    labels_list = []
    with torch.no_grad():

        for d in testloader:
            inputs = d["input_ids"].squeeze(1).to(device)
            attention_mask = d["mask"].squeeze(1).to(device)
            targets = d["targets"].to(device)
            token_type_ids =d['token_type_ids'].squeeze(1).to(device)
            outputs = model(input_ids=inputs,attention_mask=attention_mask,token_type_ids = token_type_ids)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            logits_list.append(outputs)
            labels_list.append(targets)

        logits = torch.cat(logits_list).cpu().numpy()
        labels = torch.cat(labels_list).cpu().numpy()

        ece = tfp.stats.expected_calibration_error(num_bins=10, logits=logits, labels_true=labels.astype(int), labels_predicted=np.argmax(logits,1))
    return correct / total, ece

class Wikiart_Dataset_split(Dataset):

    def __init__(self, title, targets, tokenizer, max_len, mh_confidence=None, istrain = True):
        self.title = title
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.istrain = istrain
        if self.istrain:
            self.mhc = mh_confidence
            self.title = self.title[:2500]
            self.targets = self.targets[:2500]
        else:
            self.title = self.title[2500:]
            self.targets = self.targets[2500:]
  
    def __len__(self):
        return len(self.title)
  
    def __getitem__(self, item):
        input = str(self.title[item])
        targets = self.targets[item]

        encoding = self.tokenizer.encode_plus(
          input,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=True,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',)
        ids = encoding['input_ids']
        mask = encoding['attention_mask']
        token_type_ids = encoding["token_type_ids"]
        if self.istrain:
            return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float),
            'conf':torch.tensor(self.mhc[item], dtype=torch.float)}
        else:
            return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float)}

class Wikiart_Dataset(Dataset):

    def __init__(self, title, targets, tokenizer, max_len, mh_confidence=None, istrain = True):
        self.title = title
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.istrain = istrain
        if self.istrain:
            self.mhc = mh_confidence
  
    def __len__(self):
        return len(self.title)
  
    def __getitem__(self, item):
        input = str(self.title[item])
        targets = self.targets[item]

        encoding = self.tokenizer.encode_plus(
          input,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=True,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',)
        ids = encoding['input_ids']
        mask = encoding['attention_mask']
        token_type_ids = encoding["token_type_ids"]
        if self.istrain:
            return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float),
            'conf':torch.tensor(self.mhc[item], dtype=torch.float)}
        else:
            return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float)}


def get_ad(ad):
    all_ad = []
    for i in range(len(ad)):
        per_ad = np.array(ad[i][2:-2].split(' '), dtype = np.float32)
        all_ad.append(per_ad)
    all_ad = np.array(all_ad)
    return all_ad


def main():
    seed_everything()
    warnings.filterwarnings("ignore")
    transformers.logging.set_verbosity_error()
    #standard wikiart dataloader
    df = pd.read_csv("processed_wikiart_title.csv")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=12)
    # PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
   

    #standard model 
    model = SentimentClassifier(args.num_classes)
    model = model.to(device)
    # model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=args.num_classes).cuda()

    #model for ranking cl

    if not args.split:
        model_rank = copy.deepcopy(model).to(device)
        model_rank.load_state_dict(torch.load('ckpt_split/best_bert_wikiart_split1_mcls0_mccl0_hcls0_hccl0.pth.tar'))
        model_rank.eval()
        train_set = Wikiart_Dataset(title=df_train.Title.to_numpy(),targets=df_train.label.to_numpy(),
                                tokenizer=tokenizer,max_len=args.max_length, mh_confidence=None, istrain = False)
        train_loader_rank = DataLoader(train_set, args.batch_size, shuffle =False, num_workers=2)
        args.mu, mhconf_list_all = get_threshold_confidence(model_rank, train_loader_rank, init_ratio=args.init_ratio)
    
    if args.nocl:
        args.init_ratio, args.end_epoch = 1, 0
        args.mu, mhconf_list_all = 0, torch.ones(4105)  
    elif args.mccl:
        # args.mu, mhconf_list_all = get_threshold_confidence(model_rank, train_loader_rank, init_ratio=args.init_ratio)
        args.beta = args.mu / args.end_epoch
    elif args.hccl:
        ad = get_ad(df['dist'])
        args.mu, mhconf_list_all = get_threshold_scaler_ad_sample(ad, init_ratio=args.init_ratio)
        args.beta = args.mu / args.end_epoch
    else: raise NotImplementedError

    args.mu = 0 if args.init_ratio == 1 else args.mu

    if args.split:
        train_set = Wikiart_Dataset_split(title=df_train.Title.to_numpy(),targets=df_train.label.to_numpy(),
                        tokenizer=tokenizer,max_len=args.max_length, mh_confidence= mhconf_list_all, istrain = True)

        test_set = Wikiart_Dataset_split(title=df_train.Title.to_numpy(),targets=df_train.label.to_numpy(),
                        tokenizer=tokenizer,max_len=args.max_length, mh_confidence= mhconf_list_all, istrain = False)
        
    else:
        train_set = Wikiart_Dataset(title=df_train.Title.to_numpy(),targets=df_train.label.to_numpy(),
                            tokenizer=tokenizer,max_len=args.max_length, mh_confidence= mhconf_list_all, istrain = True)
        test_set = Wikiart_Dataset(title=df_test.Title.to_numpy(),targets=df_test.label.to_numpy(),
                            tokenizer=tokenizer,max_len=args.max_length, mh_confidence= None, istrain = False)

    train_loader = DataLoader(train_set, args.batch_size, shuffle =True, num_workers=2)
    test_loader = DataLoader(test_set, args.batch_size, shuffle =False, num_workers=2)
    print('sample size for main dataloader: train:{}, test:{}'.format (len(train_set), len(test_set)))
    print("sample", train_loader.dataset[0])
    data = next(iter(train_loader))
    print("data_keys",data.keys())

    if args.mcls:
        conf_score = torch.tensor([0.8871, 0.8848, 0.8912, 0.8898, 0.7549, 0.8843]).to(device)
        mhcls_weight = args.mcls_weight
    elif args.hcls:
        conf_score = torch.tensor([0.4696, 0.5371, 0.4468, 0.4771, 0.4829, 0.5387]).to(device)
        mhcls_weight = args.hcls_weight
    else:
        conf_score = None
        mhcls_weight = None

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False, weight_decay=5e-4)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
    total_steps = len(train_loader) * args.num_epoch

    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
    criterion = CELossWith_MHCLS(classes=args.num_classes, smoothing=args.smoothing, confls=conf_score, confls_weight=mhcls_weight)
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay
    # iter_per_epoch = len(train_loader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm) 

    print('nocl:{}, mcls:{}, hcls:{}, mccl:{}, hccl:{}, init_ratio:{}, end_epoch:{}'.format(
        args.nocl, args.mcls, args.hcls, args.mccl, args.hccl, args.init_ratio, args.end_epoch))
    #training script
    best_epoch, best_acc, best_ece = 0.0, 0, np.inf

    for epoch in range(args.num_epoch):
       
        train(model, train_loader, criterion, optimizer, epoch, scheduler, args)
        
        if epoch <= args.end_epoch:
                args.mu -= args.beta
                args.mu = max(args.mu, 0)
        else:
            args.mu = 0
            
        accuracy, ece = test(model, test_loader)
        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch
            best_ece = ece
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), 'ckpt_mccl/best_bert_wikiart_split{}_mcls{}_mccl{}_hcls{}_hccl{}.pth.tar'.format(
                int(args.split), int(args.mcls), int(args.mccl), int(args.hcls), int(args.hccl)))
        print('epoch: {}  acc: {:.4f}  best epoch: {}  best acc: {:.4f} threshold: {:.4f} lr: {:.6f} best ece:{:.4f} smoothing:{:.4f}'.format(
                epoch, accuracy, best_epoch, best_acc, args.mu,optimizer.param_groups[0]['lr'], best_ece, args.smoothing ))

    ckpt_name = 'best_bert_wikiart_split{}_mcls{}_mccl{}_hcls{}_hccl{}.pth.tar'.format(
                int(args.split), int(args.mcls), int(args.mccl), int(args.hcls), int(args.hccl))
    write_csv("results_wikiart_bert.csv", ["init_prob:" + str(args.init_ratio), "epoch_iid:" + str(args.end_epoch),
                                    "threshold:{:.4f}".format(args.mu),
                                    "best_epoch:"+ str(best_epoch),
                                    "best_acc:" + str(best_acc),
                                    "ece:" + str(np.array(best_ece)),
                                    "smoothing:" + str(args.smoothing),
                                    "mcls_weight:" + str(args.mcls_weight),
                                    "hcls_weight:" + str(args.hcls_weight),
                                    "ckpt_name:" + str(ckpt_name)
                                    ])

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description='Wikiart Training')
    # parser.add_argument('--root', default='cifar-10h/data', type=str, help='learning rate')
    parser.add_argument('--split', action='store_true', help='split or full')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', default=0, type=int, help='lr scheduler')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--test_batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=50, type=int, help='epoch number')
    parser.add_argument('--num_classes', type=int, default=6, help='number classes')
    parser.add_argument('--max_length', type=int, default=30, help='max length')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--smoothing', default = 0.1, type=float, help='ls smoothing')
    parser.add_argument('--mcls', action='store_true', help='mcls or ls')
    parser.add_argument('--mcls_weight', default = -1, type=float, help='mcls weight')
    parser.add_argument('--hcls', action='store_true', help='hcls or ls')
    parser.add_argument('--hcls_weight', default = -1, type=float, help='hcls_weight')
    #parameters for curriculum learning
    parser.add_argument('--nocl', action='store_true', help='cl or not')
    parser.add_argument('--mccl', action='store_true', help='mccl or not')
    parser.add_argument('--hccl', action='store_true', help='hccl or not')
    parser.add_argument('--init_ratio', default = 0.75, type=float, help='initial prob')
    parser.add_argument('--beta', default = 0.75, type=float, help='initial prob')
    parser.add_argument('--end_epoch',default = 30, type=int, help='epoch iid')
    args = parser.parse_args()
    main()
