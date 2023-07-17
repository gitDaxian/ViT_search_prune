import math
import time

import torch
from torch import nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

from models import VGG
from utils import progress_bar, load_partial_weight, _pil_interp

from ptflops import get_model_complexity_info

from models.vit_pw import ViT, channel_selection
from models.vit_slim_pw import ViT_slim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

# 单独测试每个类精度
def test_one_class(model,transform_test):

    testset = torchvision.datasets.CIFAR10(
        # root='/home/Datasets',
        root='/home/users/xjs/DataSets',
        train=False,
        transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False,
        num_workers=0, pin_memory=True
    )

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    N_CLASSES = 10
    class_correct = list(0. for _ in range(N_CLASSES))
    class_total = list(0. for _ in range(N_CLASSES))

    model.eval()
    # test
    total_correct = 0
    total_num = 0
    start = time.time()
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            total_correct += torch.eq(pred, targets).float().sum().item()
            total_num += inputs.size(0)  # 即batch_size

            c = (pred == targets).squeeze()

            for i in range(len(targets)):
                _label = targets[i]
                class_correct[_label] += c[i].item()
                class_total[_label] += 1
        end = time.time()
        print('Acc: %.3f%% (%d/%d)' % (100. * total_correct / total_num, total_correct, total_num))
        print('Test time:{}s'.format(end - start))

        for i in range(N_CLASSES):
            print('Accuracy of %5s : %.1f%%' % (
                classes[i], 100. * class_correct[i] / class_total[i]))

# 训练
def kd(tea_net, stu_net, epochs):
    scale_size = int(math.floor(224 / 0.9))
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    trainset = torchvision.datasets.CIFAR10(
        # root='/home/DataSets',
        root='/home/users/xjs/DataSets',
        train=True,
        transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=True
    )

    vit_transform_train = transforms.Compose([
            transforms.Resize(scale_size, _pil_interp('bicubic')),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    vit_trainset = torchvision.datasets.CIFAR10(
        # root='/home/DataSets',
        root='/home/users/xjs/DataSets',
        train=True,
        transform=vit_transform_train)

    vit_trainloader = torch.utils.data.DataLoader(
        vit_trainset, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=True
    )

    optimizer = torch.optim.SGD(stu_net.parameters(),lr=1e-3,momentum=0.9,weight_decay=1e-4)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,50,eta_min=0.0003)
    # criterion = nn.CrossEntropyLoss().cuda()
    for epoch in range(epochs):
        print('\nEpoch: %d' % epoch)
        tea_net.eval()
        stu_net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, ((inputs, targets), (vit_inputs, _)) in enumerate(zip(trainloader, vit_trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            vit_inputs = vit_inputs.to(device)
            optimizer.zero_grad()

            y_teacher = tea_net(vit_inputs)
            y_student = stu_net(inputs)
            # 计算loss
            T, a = 4.0, 0.3
            lossKD = nn.KLDivLoss()
            loss_dis = lossKD(F.log_softmax(y_student / T, dim=1),
                              F.softmax(y_teacher / T, dim=1))
            lossFun = nn.CrossEntropyLoss()
            loss_stu = lossFun(y_student, targets)
            loss = (1 - a) * T * T * loss_dis + a * loss_stu
            loss = loss.cuda()
            loss.backward()
            # sparse_selection()
            optimizer.step()

            _, preds = y_student.max(1)
            correct += preds.eq(targets).sum().item()
            train_loss += loss.item()
            total += targets.size(0)
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        if epoch % 5 == 0:
            torch.save(stu_net.state_dict(),
                       'checkpoint/cifar10/vit_kd_vgg.pth')
            print("save checkpoint in epoch:{}".format(epoch))

def train(model, epochs):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    trainset = torchvision.datasets.CIFAR10(
        # root='/home/DataSets',
        root='/home/users/xjs/DataSets',
        train=True,
        transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=True
    )

    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9,weight_decay=1e-4)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,50,eta_min=0.0003)
    criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # sparse_selection()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       'checkpoint/cifar10/vgg19_init.pth')
            print("save checkpoint in epoch:{}".format(epoch))


if __name__ == '__main__':

    scale_size = int(math.floor(224 / 0.9))
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    vit_transform_test = transforms.Compose([
        transforms.Resize(scale_size, _pil_interp('bicubic')),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    vgg_transform_test = transforms.Compose([
        transforms.Resize(32),
        # transforms.Resize(224),
        transforms.ToTensor(),
        normalize
    ])
    # 定义教师模型
    tea_net = ViT(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
        qkv_bias=True
    )
    tea_net = tea_net.to(device)
    # 定义学生模型
    stu_net = VGG('VGG19')
    stu_net = stu_net.to(device)

    # 导入权重文件
    tea_model_path = "checkpoint/cifar10/advance_cs/cifar10_pruned_attn0.0_ffn0.0.pth"
    print("1 => loading teacher checkpoint '{}'".format(tea_model_path))
    weight = torch.load(tea_model_path)
    load_partial_weight(tea_net, weight)
    print("2 => test teacher:")
    # test_one_class(tea_net, vit_transform_test)
    flops, params = get_model_complexity_info(tea_net, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    print('tea flops: ', flops, 'tea params: ', params)

    # train(stu_net, 100)
    # test_one_class(stu_net, vgg_transform_test)
    stu_model_path = "checkpoint/cifar10/vit_kd_vgg.pth"
    print("1 => loading student checkpoint '{}'".format(stu_model_path))
    weight = torch.load(stu_model_path)
    stu_net.load_state_dict(weight)
    print("2 => test student:")
    test_one_class(stu_net, vgg_transform_test)
    # flops, params = get_model_complexity_info(stu_net, (3, 224, 224), as_strings=True,
    #                                           print_per_layer_stat=False)
    # print('stu flops: ', flops, 'stu params: ', params)
    # train(stu_net, 51)
    test_one_class(stu_net, vgg_transform_test)
    kd(tea_net, stu_net, 51)
    test_one_class(stu_net, vgg_transform_test)