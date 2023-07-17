import json
import math
import os
import sys
import time

import torch
import torchvision
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision.transforms as transforms

from my_dataset import MyDataSet
from utils import progress_bar, load_partial_weight, _pil_interp

from ptflops import get_model_complexity_info

from models.vit_pw import ViT
from models.vit_slim_pw import ViT_slim
from models.vit_slim_cs import ViT_slim_cs, channel_selection

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def getFileSize(filePath):
    fsize = os.path.getsize(filePath)  # 返回的是字节大小
    '''
    为了更好地显示，应该时刻保持显示一定整数形式，即单位自适应
    '''
    if fsize < 1024:
        return (round(fsize, 2), 'Byte')
    else:
        KBX = fsize / 1024
        if KBX < 1024:
            return str(round(KBX, 1)) + 'K'
        else:
            MBX = KBX / 1024
            if MBX < 1024:
                return str(round(MBX, 1)) + 'M'
            else:
                return (round(MBX / 1024), 'G')


# 测试
def tst(model):
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
        testset, batch_size=200, shuffle=False,
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
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            total_correct += torch.eq(pred, targets).float().sum().item()
            total_num += targets.size(0)  # 即batch_size

            c = (pred == targets).squeeze()
            for i in range(len(targets)):
                _label = targets[i]
                class_correct[_label] += c[i].item()
                class_total[_label] += 1

            progress_bar(batch_idx, len(testloader), 'Test Acc: %.3f%% (%d/%d)'
                         % (100. * total_correct / total_num, total_correct, total_num))

        end = time.time()
        test_acc = round(100. * total_correct / total_num, 2)

        print('Test Acc: %.3f%% (%d/%d)' % (test_acc, total_correct, total_num))
        print('Test time:{}s'.format(end - start))

        print('Accuracy of class:')
        for i in range(N_CLASSES):
            print('Accuracy of %10s : %.1f%%' % (classes[i], 100. * class_correct[i] / class_total[i]), end=" ")
        print()
    return test_acc


# 剪枝后的微调
def train(model, epochs, attn_ratio, ffn_ratio):
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
        root='/home/Datasets',
        # root='/home/users/xjs/DataSets',
        train=True,
        transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=True
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=0.0003)
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
                       'checkpoint/cifar10/advance_cs/cifar10_pruned_attn' + str(attn_ratio) + '_ffn' + str(ffn_ratio) + '.pth')
            print("save checkpoint in epoch:{}".format(epoch))


# 剪枝
def prune(attn_ratio, ffn_ratio, flag):

    # 已有该文件直接测试

    if os.path.exists('checkpoint/cifar10/advance_cs/cifar10_pruned_attn' + str(attn_ratio) + '_ffn' + str(ffn_ratio) + '.pth'):
        file_name = 'cifar10_pruned_attn' + str(attn_ratio) + '_ffn' + str(ffn_ratio) + '.pth'
        fjson = "checkpoint/cifar10/advance_cs/model_cfg_prune.json"
        # 读取原始json文件
        with open(fjson, 'r') as f:
            content = json.load(f)
        cfg_prune = content[file_name]
        # 定义模型导入权重文件
        model = ViT_slim_cs(
            image_size=224,
            patch_size=16,
            num_classes=10,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
            emb_dropout=0.1,
            qkv_bias=True,
            cfg=cfg_prune)  # 导入之前的文件
        model.to(device)

        # 导入权重文件
        print("loading already checkpoint: {}".format(file_name))
        weight = torch.load('checkpoint/cifar10/advance_cs/'+file_name)
        load_partial_weight(model, weight)

        # train(model, 6, attn_ratio, ffn_ratio)

        # 微调后测试
        print("test : ")
        pruned_acc = tst(model)
        file_name = 'cifar10_pruned_attn' + str(attn_ratio) + '_ffn' + str(ffn_ratio) + '.pth'
        file_path = 'checkpoint/cifar10/advance_cs/' + file_name
        torch.save(model.state_dict(), file_path)
        return pruned_acc


    # 读取前置权重进行迭代式剪枝
    if flag == 0: # 剪枝ffn_ratio
        pw_ffn_ratio = ffn_ratio-0.1 if len(str(ffn_ratio).split(".")[1]) == 1 else int(ffn_ratio*10)/10
        pw_ffn_ratio = round(pw_ffn_ratio, 1)
        ffn_str = str(pw_ffn_ratio)
        file_name = 'cifar10_pruned_attn' + str(attn_ratio) + '_ffn' + ffn_str + '.pth'
    else: # 剪枝attn_ratio
        pw_attn_ratio = attn_ratio - 0.1 if len(str(attn_ratio).split(".")[1]) == 1 else int(attn_ratio * 10) / 10
        pw_attn_ratio = round(pw_attn_ratio, 1)
        attn_str = str(pw_attn_ratio)
        file_name = 'cifar10_pruned_attn' + attn_str + '_ffn' + str(ffn_ratio) + '.pth'

    cfg_prune = []
    if os.path.exists('checkpoint/cifar10/advance_cs/'+file_name):
        fjson = "checkpoint/cifar10/advance_cs/model_cfg_prune.json"
        # 读取原始json文件
        with open(fjson, 'r') as f:
            content = json.load(f)
        cfg_prune = content[file_name]

    # 定义模型导入权重文件
    model = ViT_slim_cs(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
        qkv_bias=True,
        cfg=cfg_prune)  # 导入之前的文件
    model.to(device)

    # 导入权重文件
    print("loading checkpoint: {}".format(file_name))
    weight = torch.load('checkpoint/cifar10/advance_cs/'+file_name)
    load_partial_weight(model, weight)

    # 计算总通道数量total
    attn_total = 0
    ffn_total = 0
    for k, m in enumerate(model.modules()):
        if isinstance(m, channel_selection):
            if k in [15, 37, 59, 81, 103, 125, 147, 169, 191, 213, 235, 257]:
                attn_total += m.indexes.data.shape[0]
            else:
                ffn_total += m.indexes.data.shape[0]
    print("attn_total: ", attn_total)
    print("ffn_total: ", ffn_total)

    # 在当前的权重文件上真实的剪枝比例
    if flag == 0:
        real_attn_ratio = 0.0
        real_ffn_ratio = (ffn_total-36864*(1-ffn_ratio))/ffn_total
    else:
        real_attn_ratio = (attn_total-9216*(1-attn_ratio))/attn_total
        real_ffn_ratio = 0.0

    real_attn_ratio = round(real_attn_ratio, 2)
    real_ffn_ratio = round(real_ffn_ratio, 2)

    # 取出剪枝参数
    attn_bn = torch.zeros(attn_total)
    ffn_bn = torch.zeros(ffn_total)
    attn_index = 0
    ffn_index = 0
    for k, m in enumerate(model.modules()):
        if isinstance(m, channel_selection):
            if k in [15, 37, 59, 81, 103, 125, 147, 169, 191, 213, 235, 257]:
                size = m.indexes.data.shape[0]  # 768
                attn_bn[attn_index:(attn_index + size)] = m.indexes.data.abs().clone()
                attn_index += size
            else:
                size = m.indexes.data.shape[0]  # 3072
                ffn_bn[ffn_index:(ffn_index + size)] = m.indexes.data.abs().clone()
                ffn_index += size

    attn_y, _ = torch.sort(attn_bn)
    ffn_y, _ = torch.sort(ffn_bn)
    attn_thre_index = int(attn_total * real_attn_ratio)  # 阈值索引
    if attn_thre_index != 0:
        attn_thre = attn_y[attn_thre_index-1]  # attn阈值
    else:
        attn_thre = attn_y[0]

    ffn_thre_index = int(ffn_total * real_ffn_ratio)  # 阈值索引
    if ffn_thre_index != 0:
        ffn_thre = ffn_y[ffn_thre_index-1]  # attn阈值
    else:
        ffn_thre = ffn_y[0]
    print("attn_ratio: {},real_attn_ratio: {}".format(attn_ratio, real_attn_ratio))
    print("ffn_ratio: {},real_ffn_ratio: {}".format(ffn_ratio, real_ffn_ratio))

    # 计算剪枝数量、剩余通道、mask列表
    pruned_attn = 0
    pruned_ffn = 0
    cfg = []
    cfg_mask = []

    for k, m in enumerate(model.modules()):
        if isinstance(m, channel_selection):
            if k in [15, 37, 59, 81, 103, 125, 147, 169, 191, 213, 235, 257]:  # encoder block中的MSA模块
                weight_copy = m.indexes.data.abs().clone()
                mask = weight_copy.gt(attn_thre).float().cuda()
                weight_copy_y, weight_copy_i = torch.sort(weight_copy, descending=True)
                aux_index = (weight_copy_y <= attn_thre).nonzero()
                i = 0
                while (torch.sum(mask) % 12 != 0):  # 必须满足heads整除
                    # print(torch.sum(mask))
                    mask[weight_copy_i[aux_index[i]]] = 1
                    i += 1
                pruned_attn = pruned_attn + mask.shape[0] - torch.sum(mask)  # 记录剪枝数量
            else:
                weight_copy = m.indexes.data.abs().clone()
                mask = weight_copy.gt(ffn_thre).float().cuda()
                pruned_ffn = pruned_ffn + mask.shape[0] - torch.sum(mask)  # 记录剪枝数量
            m.indexes.data.mul_(mask)  # 掩膜操作
            cfg.append(int(torch.sum(mask)))  # 剩余通道
            cfg_mask.append(mask.clone())  # mask列表
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))

    pruned_attn_ratio = pruned_attn / attn_total  # 剪枝比例
    pruned_ffn_ratio = pruned_ffn / ffn_total  # 剪枝比例

    cfg_prune = []
    for i in range(len(cfg)):
        if i % 2 != 0:
            cfg_prune.append([cfg[i - 1], cfg[i]])  # 一个block中的通道cfg成对加入
    print("pruned_attn_ratio:", pruned_attn_ratio) # 真正的剪枝比例
    print("pruned_ffn_ratio:", pruned_ffn_ratio)
    print("pre-processing successful!")
    print(cfg_prune)

    # 定义一个新的模型结构复制参数(含channel_selection层)
    newmodel = ViT_slim_cs(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
        qkv_bias=True,
        cfg=cfg_prune)  # 根据上面得到的剪枝cfg_prune重新定义网络
    newmodel.to(device)
    # model_dict = model.state_dict()
    # 参数移植
    newmodel_dict = newmodel.state_dict().copy()
    i = 0
    newdict = {}
    for k, v in model.state_dict().items():
        if 'to_qkv' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            qkv_mask = torch.cat((cfg_mask[i], cfg_mask[i], cfg_mask[i]))
            idx = np.squeeze(np.argwhere(np.asarray(qkv_mask.cpu().numpy())))
            newdict[k] = v[idx.tolist()].clone()
        elif 'to_out.0.weight' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            newdict[k] = v[:, idx.tolist()].clone()
        elif '0.fn.fn.select.indexes' in k:
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            newdict[k] = v[idx.tolist()].clone()
            i = i + 1

        elif '1.fn.fn.select.indexes' in k:
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            newdict[k] = v[idx.tolist()].clone()
        elif 'net.0.weight' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            newdict[k] = v[idx.tolist()].clone()
        elif 'net.0.bias' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            newdict[k] = v[idx.tolist()].clone()
        elif 'net2.0.weight' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            newdict[k] = v[:, idx.tolist()].clone()
            i = i + 1

        elif k in newmodel.state_dict():
            newdict[k] = v

    newmodel_dict.update(newdict)
    newmodel.load_state_dict(newmodel_dict)
    print("load new dict successfully!")

    # 剪枝后微调
    train(newmodel, 16, attn_ratio, ffn_ratio)

    # print("pruned model: ")
    # flops, params = get_model_complexity_info(newmodel, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    # print('flops: ', flops, 'params: ', params)

    # 微调后测试
    print("after pruning: ")
    pruned_acc = tst(newmodel)
    file_name = 'cifar10_pruned_attn' + str(attn_ratio) + '_ffn' + str(ffn_ratio) + '.pth'
    file_path = 'checkpoint/cifar10/advance_cs/' + file_name
    torch.save(newmodel.state_dict(), file_path)


    # 更新JSON文件
    fjson = "checkpoint/cifar10/advance_cs/model_cfg_prune.json"
    # 读取原始json文件
    with open(fjson, 'r') as f:
        content = json.load(f)
    # 更新字典dict
    axis = {file_name: cfg_prune}
    content.update(axis)
    # 写入
    with open(fjson, 'w') as f_new:
        json.dump(content, f_new)
    print("update JSON file successfully!")

    # 定义一个新的模型结构复制参数(不含channel_selection层)
    real_model = ViT_slim(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
        qkv_bias=True,
        cfg=cfg_prune)  # 根据上面得到的剪枝cfg_prune重新定义网络
    real_model.to(device)

    # 参数移植
    real_model_dict = real_model.state_dict().copy()
    i = 0
    real_dict = {}
    for k, v in model.state_dict().items():
        if 'net.0.weight' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            real_dict[k] = v[idx.tolist()].clone()
        elif 'net.0.bias' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            real_dict[k] = v[idx.tolist()].clone()
        elif 'to_qkv' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            qkv_mask = torch.cat((cfg_mask[i], cfg_mask[i], cfg_mask[i]))
            idx = np.squeeze(np.argwhere(np.asarray(qkv_mask.cpu().numpy())))
            real_dict[k] = v[idx.tolist()].clone()
        elif 'net2.0.weight' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            real_dict[k] = v[:, idx.tolist()].clone()
            i = i + 1
        elif 'to_out.0.weight' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            real_dict[k] = v[:, idx.tolist()].clone()
            i = i + 1

        elif k in real_model.state_dict():
            real_dict[k] = v

    real_model_dict.update(real_dict)
    real_model.load_state_dict(real_model_dict)
    print("load really dict successfully!")

    flops, params = get_model_complexity_info(real_model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    print('flops: ', flops, 'params: ', params)

    # file_name = 'imagenet_pruned_attn' + str(attn_ratio) + '_ffn' + str(ffn_ratio) + '.pth'
    file_path = 'checkpoint/cifar10/advance/' + file_name
    torch.save(real_model.state_dict(), file_path)

    print(
        'cifar10_pruned_attn' + str(attn_ratio) + '_ffn' + str(ffn_ratio) + '.pth' + '==>' + getFileSize(file_path))

    return pruned_acc



if __name__ == '__main__':


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
        qkv_bias=True,
    )
    model = model.to(device)

    # 导入权重文件
    model_path = "checkpoint/cifar10/advance_cs/cifar10_pruned_attn0.0_ffn0.0.pth"
    print("loading checkpoint '{}'".format(model_path))
    weight = torch.load(model_path)
    load_partial_weight(model, weight)

    # 先进行测试
    orgin_acc = tst(model)
    # 变量初始化
    # percentage = 0.90  # 剪枝目标为原精度的百分比
    # target = round(orgin_acc * percentage, 2)
    target = 95.71
    error = 0.2  # 误差
    print("target region:[{}, {}]".format(target-error, target+error))
    attn_ratio = 0.0  # 模块初始剪枝比例
    ffn_ratio = 0.0
    closed_acc = orgin_acc  # 记录搜索过程中的最接近值
    closed_attn_ratio = attn_ratio
    closed_ffn_ratio = ffn_ratio
    length = 0.1  # 候选区间查找步幅
    bin_search_thre = 0.01  # 二分查找阈值
    flag = 0  # 是否已经查找到值
    i = 0 # 查找次数

    # 1.ffn_ratio的候选区间查找
    print("1.start search candidate region of ffn_ratio-------------------------------------------------")
    pruned_acc = orgin_acc
    while ffn_ratio < 1:
        if pruned_acc < target - error:
            break
        elif pruned_acc > target + error:
            ffn_ratio += length
            ffn_ratio = round(ffn_ratio, 1)
            if ffn_ratio != 1.0:
                pruned_acc = prune(attn_ratio, ffn_ratio, 0)
                i += 1
                print("Search times:{} ------------------------------------------------------".format(i))
        else:
            flag = 1
            break
    if flag == 1:
        print("Successfully Search!->search times:{},best attn_ratio:{},best ffn_ratio:{},pruned_acc:{}".format(i,
                                                                                                                attn_ratio,
                                                                                                                ffn_ratio,
                                                                                                                pruned_acc))
        sys.exit(0)
    if ffn_ratio == 1.0:
        ffn_ratio -= length
        ffn_ratio = round(ffn_ratio, 1)

    # 2.ffn_ratio的二分查找
    print("2.start binary search of ffn_ratio-----------------------------------------------------------")
    # low = ffn_ratio if ffn_ratio == 0.0 else ffn_ratio - 0.1
    # high = ffn_ratio
    low = 0.90
    high = 0.95
    mid = 0.0
    while low <= high and high - low >= bin_search_thre:
        mid = round((low + high) / 2, 2)
        pruned_acc = prune(attn_ratio, mid, 0)
        i += 1
        print("Search times:{} ------------------------------------------------------".format(i))
        # 更新最接近值
        if abs(closed_acc - target) > abs(pruned_acc - target):
            closed_acc = pruned_acc
            closed_attn_ratio = attn_ratio
            closed_ffn_ratio = mid
        if pruned_acc > target + error:
            low = mid + 0.01
        elif pruned_acc < target + error:
            high = mid - 0.01
        else:
            flag = 1
            break
    ffn_ratio = mid

    if flag == 1:
        print("Successfully Search!->search times:{},best attn_ratio:{},best ffn_ratio:{},pruned_acc:{}".format(i,
                                                                                                                attn_ratio,
                                                                                                                ffn_ratio,
                                                                                                                pruned_acc))
        sys.exit(0)

    # 3.attn_ratio的候选区间查找
    print("3.start search candidate region of attn_ratio-------------------------------------------------")
    while attn_ratio < 1:
        if pruned_acc < target - error:
            break
        elif pruned_acc > target + error:
            attn_ratio += length
            attn_ratio = round(attn_ratio, 1)
            if attn_ratio != 1.0:
                pruned_acc = prune(attn_ratio, ffn_ratio, 1)
                i += 1
                print("Search times:{} ------------------------------------------------------".format(i))
        else:
            flag = 1
            break
    if flag == 1:
        print("Successfully Search!->search times:{},best attn_ratio:{},best ffn_ratio:{},pruned_acc:{}".format(i,
                                                                                                                attn_ratio,
                                                                                                                ffn_ratio,
                                                                                                                pruned_acc))
        sys.exit(0)

    # 4.attn_ratio的二分查找
    print("4.start binary search of attn_ratio-----------------------------------------------------------")
    low = attn_ratio if attn_ratio == 0 else attn_ratio - 0.1
    high = attn_ratio
    mid = 0.0
    while low <= high and high - low >= bin_search_thre:
        mid = round((low + high) / 2, 2)
        pruned_acc = prune(mid, ffn_ratio, 1)
        i += 1
        print("Search times:{} ------------------------------------------------------".format(i))
        # 更新最接近值
        if abs(closed_acc - target) > abs(pruned_acc - target):
            closed_acc = pruned_acc
            closed_attn_ratio = mid
            closed_ffn_ratio = ffn_ratio
        if pruned_acc > target + error:
            low = mid + 0.01
        elif pruned_acc < target + error:
            high = mid - 0.01
        else:
            flag = 1
            break

    if flag == 1:
        print("Successfully Search!->search times:{},best attn_ratio:{},best ffn_ratio:{},pruned_acc:{}".format(i,attn_ratio, ffn_ratio,
                                                                                                pruned_acc))
    else:
        print("No Search!->search times:{},closed attn_ratio:{},closed ffn_ratio:{},pruned_acc:{}".format(i, closed_attn_ratio,
                                                                                          closed_ffn_ratio, closed_acc))
