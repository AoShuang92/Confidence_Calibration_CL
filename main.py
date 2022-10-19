import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler
import os
import argparse
import copy
import random
from PIL import Image
import numpy as np
import csv
import tensorflow_probability as tfp

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


class CELossWith_MCLS(torch.nn.Module):
    
    def __init__(self, classes=1000, smoothing=0.1, ignore_index=-1, confls=None, confls_weight=None):
        super(CELossWith_MCLS, self).__init__()
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
        for batch_idx, (inputs, targets) in enumerate(trainloader_rank):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_rank(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            conf = F.softmax(outputs, dim=1)
            conf_list = [conf_each[targets[i]] for i, conf_each in enumerate(conf)]
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

def train(model, trainloader, criterion, optimizer, epoch, warmup_scheduler, args):
    model.train()
    
    for batch_idx, (inputs, targets, conf_list) in enumerate(trainloader):
        inputs, targets, conf_list = inputs.to(device), targets.to(device), conf_list.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)    
        loss = criterion(outputs[conf_list>=args.mu], targets[conf_list>=args.mu])
        loss.backward()
        optimizer.step()
        if epoch <= args.warm:
            warmup_scheduler.step()
        
        
def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            logits_list.append(outputs)
            labels_list.append(targets)

        logits = torch.cat(logits_list).cpu().numpy()
        labels = torch.cat(labels_list).cpu().numpy()
        ece = tfp.stats.expected_calibration_error(10, logits=logits, labels_true=labels, labels_predicted=np.argmax(logits,1))
    return correct / total, ece

class CIFAR10H_split(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=False, transform=None, target_transform=None,
                 download=False, mh_confidence=None, istrain = False):
        super(CIFAR10H_split, self).__init__(root, train, transform, target_transform, download) 
        self.transform = transform
        self.target_transform = target_transform
        self.istrain = istrain
        if self.istrain:
            self.mhc = mh_confidence
            self.data = self.data[:8000]
            self.targets = self.targets[:8000]
        else:
            self.data = self.data[8000:]
            self.targets = self.targets[8000:]
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.istrain:
            return img, target, self.mhc[index]
        else:
            return img, target

class CIFAR10H(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=False, transform=None, target_transform=None,
                 download=False, mh_confidence=None, istrain = False):
        super(CIFAR10H, self).__init__(root, train, transform, target_transform, download) 
        self.transform = transform
        self.target_transform = target_transform
        self.istrain = istrain
        if self.istrain:
            self.mhc = mh_confidence

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.istrain:
            return img, target, self.mhc[index]
        else:
            return img, target


def main():
    seed_everything()

    #standard model 
    model = models.resnet34(pretrained=True).to(device)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model = model.to(device)

    #model for ranking cl
    mean_cifar10, std_cifar10 = (0.491, 0.482, 0.446), (0.247, 0.243, 0.261)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                transforms.Normalize(mean_cifar10, std_cifar10), ])
    transform_test = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(mean_cifar10, std_cifar10),])

    if not args.split:
        model_rank = copy.deepcopy(model).to(device)
        model_rank.load_state_dict(torch.load('ckpt_split/best_model_cifar10h_split1_mcls0_mccl0_mcls0_mccl0.pth.tar'))
        model_rank.eval()
        train_dataset_rank = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        train_loader_rank = torch.utils.data.DataLoader(train_dataset_rank, batch_size=args.test_batch_size, 
                shuffle=False, num_workers=2)
        print('sample size for ranking dataloader:', len(train_dataset_rank))
        args.mu, mhconf_list_all = get_threshold_confidence(model_rank, train_loader_rank, init_ratio=args.init_ratio)
    
    if args.nocl:
        args.init_ratio, args.end_epoch = 1, 0
        args.mu, mhconf_list_all = 0, torch.ones(10000)    
    elif args.mccl:
        # args.mu, mhconf_list_all = get_threshold_confidence(model_rank, train_loader_rank, init_ratio=args.init_ratio)
        args.beta = args.mu / args.end_epoch
    elif args.hccl:
        ad = np.load(os.path.join(args.root,'cifar10h-probs.npy'))
        args.mu, mhconf_list_all = get_threshold_scaler_ad_sample(ad, init_ratio=args.init_ratio)
        args.beta = args.mu / args.end_epoch
    else: raise NotImplementedError

    args.mu = 0 if args.init_ratio == 1 else args.mu

    #standard cifar10h dataloader
    if args.split:
        train_dataset = CIFAR10H_split(root=args.root, train=False, download=True, transform=transform_train, 
                            mh_confidence=mhconf_list_all, istrain = True)
        test_dataset = CIFAR10H_split(root=args.root, train=False, download=True, transform=transform_test)
    else:
        train_dataset = CIFAR10H(root=args.root, train=False, download=True, transform=transform_train, 
                            mh_confidence=mhconf_list_all, istrain = True)
        test_dataset = CIFAR10H(root=args.root, train=True, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    print('sample size for main dataloader: train:{}, test:{}'.format (len(train_dataset), len(test_dataset)))
    #optimizer, criterion and scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False, weight_decay=5e-4)
    
    
    if args.mcls:
        conf_score = torch.tensor([0.7786, 0.7868, 0.7243, 0.6908, 0.7675, 0.7220, 0.7791, 0.7759, 0.7832,
        0.7925]).to(device)
        mhcls_weight = args.mcls_weight
    elif args.hcls:
        conf_score = torch.tensor([0.9520, 0.9720, 0.9481, 0.9150, 0.9050, 0.9494, 0.9618, 0.9794, 0.9730,
        0.9677]).to(device)
        mhcls_weight = args.hcls_weight
    else:
        conf_score = None
        mhcls_weight = None 

    criterion = CELossWith_MCLS(classes=args.num_classes, smoothing=args.smoothing, confls=conf_score, confls_weight=mhcls_weight)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm) 

    print('nocl:{}, mcls:{}, hcls:{}, mccl:{}, hccl:{}, init_ratio:{}, end_epoch:{}'.format(
        args.nocl, args.mcls, args.hcls, args.mccl, args.hccl, args.init_ratio, args.end_epoch))
    #training script
    best_epoch, best_acc, best_ece = 0.0, 0, np.inf

    for epoch in range(1, args.num_epoch + 1):
        if epoch > args.warm:
                train_scheduler.step()
       
        train(model, train_loader, criterion, optimizer, epoch, warmup_scheduler, args)
        
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
            torch.save(best_model.state_dict(), 'ckpt_mccl/best_model_cifar10h_split{}_mcls{}_mccl{}_hcls{}_mccl{}.pth.tar'.format(
                int(args.split), int(args.mcls), int(args.mccl), int(args.hcls), int(args.hccl)))
        print('epoch: {}  acc: {:.4f}  best epoch: {}  best acc: {:.4f} threshold: {:.4f} lr: {:.4f} best ece:{:.4f} smoothing:{:.4f}'.format(
                epoch, accuracy, best_epoch, best_acc, args.mu,optimizer.param_groups[0]['lr'], best_ece, args.smoothing ))

    ckpt_name = 'best_model_cifar10h_split{}_mcls{}_mccl{}_hcls{}_hccl{}.pth.tar'.format(
                int(args.split), int(args.mcls), int(args.mccl), int(args.hcls), int(args.hccl))
    write_csv("results.csv", ["init_prob:" + str(args.init_ratio), "epoch_iid:" + str(args.end_epoch),
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
    parser = argparse.ArgumentParser(description='CIFAR-10H Training')
    parser.add_argument('--root', default='cifar-10h/data', type=str, help='learning rate')
    parser.add_argument('--split', action='store_true', help='split or full')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', default=0, type=int, help='lr scheduler')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--test_batch_size', default=2048, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=200, type=int, help='epoch number')
    parser.add_argument('--num_classes', type=int, default=10, help='number classes')
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