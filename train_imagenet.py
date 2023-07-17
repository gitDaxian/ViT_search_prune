# -*- coding: utf-8 -*-
'''

Train ImageNet with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
# import pandas as pd
import csv

from models import *
from models.vit import ViT, channel_selection
from my_dataset import MyDataSet
from utils import progress_bar

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parsers
parser = argparse.ArgumentParser(description='PyTorch IMAGENET Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='add image augumentations')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='64')
parser.add_argument('--n_epochs', type=int, default='100')
parser.add_argument('--patch', default='16', type=int)
parser.add_argument('--cos', action='store_true', help='Train with cosine annealing scheduling')
args = parser.parse_args()

if args.cos:
    from warmup_scheduler import GradualWarmupScheduler
if args.aug:
    import albumentations
bs = int(args.bs)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomCrop(224),#随机裁剪
    transforms.RandomHorizontalFlip(),#随机水平翻转
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])

data_root = "/home/Datasets/mini-imagenet"
json_path = "/home/Datasets/mini-imagenet/classes_name.json"
trainset = MyDataSet(root_dir=data_root,
                              csv_name="new_train.csv",
                              json_path=json_path,
                              transform=transform_train)

testset = MyDataSet(root_dir=data_root,
                              csv_name="new_test.csv",
                              json_path=json_path,
                              transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)

testloader = torch.utils.data.DataLoader(testset,batch_size=64,shuffle=False)



# Model
print('==> Building model..')

net = ViT(
    image_size=224,
    patch_size=args.patch,
    num_classes=100,
    dim=768,  # 512
    depth=12,
    heads=12,
    mlp_dim=768,
    dropout=0.1,
    emb_dropout=0.1
)

net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net) # make parallel
#     cudnn.benchmark = True
# cudnn.benchmark = True


# if args.resume:
# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/{}-{}-{}.t7'.format(args.net, args.patch,"imagenet"))
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()
# reduce LR on Plateau
if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
elif args.opt == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)

if not args.cos:
    from torch.optim import lr_scheduler

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3 * 1e-5,
                                               factor=0.1)
else:
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs - 1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)


def sparse_selection():
    s = 1e-4
    for m in net.modules():
        if isinstance(m, channel_selection):
            m.indexes.grad.data.add_(s * torch.sign(m.indexes.data))  # L1


##### Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        sparse_selection()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return train_loss / (batch_idx + 1)


##### Validation
import time


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.net + '-{}-{}.t7'.format(args.patch,"imagenet"))
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc


list_loss = []
list_acc = []
for epoch in range(start_epoch, args.n_epochs):
    trainloss = train(epoch)
    val_loss, acc = test(epoch)

    if args.cos:
        scheduler.step(epoch - 1)

    list_loss.append(val_loss)
    list_acc.append(acc)

    # write as csv for analysis
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss)
        writer.writerow(list_acc)
        # print(list_loss)


