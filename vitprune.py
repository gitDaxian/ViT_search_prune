import importlib
import math
import os
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
import argparse

import torchvision
import torchvision.transforms as transforms
from utils import progress_bar, load_partial_weight, _pil_interp

from ptflops import get_model_complexity_info

from models.vit_pw import ViT, channel_selection
from models.vit_slim_pw import ViT_slim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

# 单独测试每个类精度
def test_one_class(model):
    scale_size = int(math.floor(224 / 0.9))
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    transform_test = transforms.Compose([
        transforms.Resize(scale_size, _pil_interp('bicubic')),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    testset = torchvision.datasets.CIFAR10(
        root='/home/Datasets',
        # root='/home/users/xjs/DataSets',
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

# 剪枝后的微调
def train(model, epochs, percent):
    scale_size = int(math.floor(224 / 0.9))
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    transform_train = transforms.Compose([
            transforms.Resize(scale_size, _pil_interp('bicubic')),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
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
                       'checkpoint/cifar10/vtp/pruned_' + str(percent) + '.pth')
            print("save checkpoint in epoch:{}".format(epoch))


def getFileSize(filePath):
    fsize = os.path.getsize(filePath)	# 返回的是字节大小
    '''
    为了更好地显示，应该时刻保持显示一定整数形式，即单位自适应
    '''
    if fsize < 1024:
    	return(round(fsize,2),'Byte')
    else:
    	KBX = fsize/1024
    	if KBX < 1024:
    		return str(round(KBX,1))+'K'
    	else:
    		MBX = KBX /1024
    		if MBX < 1024:
    			return str(round(MBX,1))+'M'
    		else:
    			return(round(MBX/1024),'G')
def run(path):
    model = importlib.import_module('.vgg', package='{}'.format(path))
    return model


if __name__ == '__main__':

    path = 'help.models'
    m = run(path)
    model = m.VGG('VGG19')
    print(model)

    # 定义模型导入权重文件
    model = ViT(
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
    model = model.to(device)

    # 导入权重文件
    model_path = "checkpoint/cifar10/advance_cs/cifar10_pruned_attn0.0_ffn0.0.pth"
    print("1 => loading checkpoint '{}'".format(model_path))
    weight = torch.load(model_path)
    load_partial_weight(model, weight)
    test_one_class(model)
    # flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)

    # total = 0
    # for m in model.modules():
    #     if isinstance(m, channel_selection):
    #         total += m.indexes.data.shape[0]
    #
    # bn = torch.zeros(total)
    # index = 0
    # for m in model.modules():
    #     if isinstance(m, channel_selection):
    #         size = m.indexes.data.shape[0]
    #         bn[index:(index + size)] = m.indexes.data.abs().clone()
    #         index += size
    #
    # percent = 0.4
    # print(percent)
    # y, i = torch.sort(bn)
    # thre_index = int(total * percent)
    # thre = y[thre_index]
    #
    #
    # # 计算剪枝数量、剩余通道、mask列表
    # pruned = 0
    # cfg = []
    # cfg_mask = []
    # for k, m in enumerate(model.modules()):
    #     if isinstance(m, channel_selection):
    #         # print(k)
    #         # print(m)
    #         if k in [15, 37, 59, 81, 103, 125, 147, 169, 191, 213, 235, 257]:
    #             weight_copy = m.indexes.data.abs().clone()
    #             mask = weight_copy.gt(thre).float().cuda()
    #             thre_ = thre.clone()
    #             while (torch.sum(mask) % 12 != 0):  # heads
    #                 thre_ = thre_ - 0.0001
    #                 mask = weight_copy.gt(thre_).float().cuda()
    #         else:
    #             weight_copy = m.indexes.data.abs().clone()
    #             mask = weight_copy.gt(thre).float().cuda()
    #         pruned = pruned + mask.shape[0] - torch.sum(mask)
    #         m.indexes.data.mul_(mask)
    #         cfg.append(int(torch.sum(mask)))
    #         cfg_mask.append(mask.clone())
    #         print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
    #               format(k, mask.shape[0], int(torch.sum(mask))))
    #
    # pruned_ratio = pruned / total
    # print('Pre-processing Successful!')
    # print(cfg)
    #
    # cfg_prune = []
    # for i in range(len(cfg)):
    #     if i % 2 != 0:
    #         cfg_prune.append([cfg[i - 1], cfg[i]])
    #
    # 定义一个新的模型结构复制参数
    # newmodel = ViT_slim(
    #     image_size=224,
    #     patch_size=16,
    #     num_classes=10,
    #     dim=768,
    #     depth=12,
    #     heads=12,
    #     mlp_dim=3072,
    #     dropout=0.1,
    #     emb_dropout=0.1,
    #     qkv_bias=True,
    #     cfg=cfg_prune)  # 根据上面得到的剪枝cfg_prune重新定义网络
    # newmodel.to(device)
    #
    # # 参数移植
    # newmodel_dict = newmodel.state_dict().copy()
    # i = 0
    # newdict = {}
    # for k, v in model.state_dict().items():
    #     if 'net.0.weight' in k:
    #         # print(k)
    #         # print(v.size())
    #         # print('----------')
    #         idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
    #         newdict[k] = v[idx.tolist()].clone()
    #     elif 'net.0.bias' in k:
    #         # print(k)
    #         # print(v.size())
    #         # print('----------')
    #         idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
    #         newdict[k] = v[idx.tolist()].clone()
    #     elif 'to_qkv' in k:
    #         # print(k)
    #         # print(v.size())
    #         # print('----------')
    #         qkv_mask = torch.cat((cfg_mask[i], cfg_mask[i], cfg_mask[i]))
    #         idx = np.squeeze(np.argwhere(np.asarray(qkv_mask.cpu().numpy())))
    #         newdict[k] = v[idx.tolist()].clone()
    #     elif 'net2.0.weight' in k:
    #         # print(k)
    #         # print(v.size())
    #         # print('----------')
    #         idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
    #         newdict[k] = v[:, idx.tolist()].clone()
    #         i = i + 1
    #     elif 'to_out.0.weight' in k:
    #         # print(k)
    #         # print(v.size())
    #         # print('----------')
    #         idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
    #         newdict[k] = v[:, idx.tolist()].clone()
    #         i = i + 1
    #
    #     elif k in newmodel.state_dict():
    #         newdict[k] = v
    #
    # newmodel_dict.update(newdict)
    # newmodel.load_state_dict(newmodel_dict)
    # print("3 => load new dict successfully!")
    #
    # # 剪枝后微调
    # train(newmodel, 16, percent)
    # print("4 => pruned model: ")
    # flops, params = get_model_complexity_info(newmodel, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    # print('flops: ', flops, 'params: ', params)
    #
    # # 微调后测试并保存
    # print("5 => after pruning: ")
    # test_one_class(newmodel)
    # torch.save(newmodel.state_dict(),
    #            'checkpoint/cifar10/vtp/pruned_' + str(percent) + '.pth')
    # filePath = 'checkpoint/cifar10/vtp/pruned_' + str(percent) + '.pth'
    # print('pruned_' + str(percent) + '.pth'+'   '+getFileSize(filePath))

    # cfg_prune = [[696, 2077], [732, 2047], [732, 2071], [444, 2008], [744, 2066], [756, 2120],
    #              [504, 2157], [660, 2258], [720, 2420], [552, 1730], [756, 892],[696, 517]]
    # # 定义一个新的模型结构复制参数
    # model = ViT_slim(
    #     image_size=224,
    #     patch_size=16,
    #     num_classes=10,
    #     dim=768,
    #     depth=12,
    #     heads=12,
    #     mlp_dim=3072,
    #     dropout=0.1,
    #     emb_dropout=0.1,
    #     qkv_bias=True,
    #     cfg=cfg_prune)  # 根据上面得到的剪枝cfg_prune重新定义网络
    # model.to(device)
    #
    # flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    # print('flops: ', flops, 'params: ', params)
    #
    # # 导入权重文件
    # model_path = "checkpoint/cifar10/vtp/pruned_0.4.pth"
    # print("1 => loading checkpoint '{}'".format(model_path))
    # weight = torch.load(model_path)
    # load_partial_weight(model, weight)
    # test_one_class(model)





