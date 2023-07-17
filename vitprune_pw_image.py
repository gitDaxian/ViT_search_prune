import math
import os
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

from my_dataset import MyDataSet
from utils import progress_bar, load_partial_weight, _pil_interp

from ptflops import get_model_complexity_info

from models.vit_pw import ViT, channel_selection
from models.vit_slim_pw import ViT_slim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

# 测试
def tst(model):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    data_root = "/home/users/xjs/DataSets/mini-imagenet"
    json_path = "/home/users/xjs/DataSets/mini-imagenet/classes_name.json"

    testset = MyDataSet(root_dir=data_root,
                        csv_name="new_test.csv",
                        json_path=json_path,
                        transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


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

        end = time.time()
        print('Acc: %.3f%% (%d/%d)' % (100. * total_correct / total_num, total_correct, total_num))
        print('Test time:{}s'.format(end - start))


# 剪枝后的微调
def train(model, epochs, attn_percent, ffn_percent):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomCrop(224),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        normalize
    ])
    data_root = "/home/users/xjs/DataSets/mini-imagenet"
    json_path = "/home/users/xjs/DataSets/mini-imagenet/classes_name.json"
    trainset = MyDataSet(root_dir=data_root,
                         csv_name="new_train.csv",
                         json_path=json_path,
                         transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

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
                       'checkpoint/imagenet_pruned_attn' + str(attn_percent) + '_ffn' + str(ffn_percent) + '.pth')
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

if __name__ == '__main__':
    # 定义模型导入权重文件
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=100,
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
    model_path = "checkpoint/imagenet_pruned_attn0_ffn0.pth"
    print("1 => loading checkpoint '{}'".format(model_path))
    weight = torch.load(model_path)
    load_partial_weight(model, weight)

    tst(model)
    # train(model,11,0,0)
    # flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)

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

    # 根据剪枝比例计算阈值
    attn_percent = 0.05
    ffn_percent = 0.95

    attn_y, _ = torch.sort(attn_bn)
    ffn_y, _ = torch.sort(ffn_bn)
    attn_thre_index = int(attn_total * attn_percent)  # 阈值索引
    attn_thre = attn_y[attn_thre_index]  # attn阈值

    ffn_thre_index = int(ffn_total * ffn_percent)  # 阈值索引
    ffn_thre = ffn_y[ffn_thre_index]  # ffn阈值
    print("attn_percent: {}".format(attn_percent))
    print("ffn_percent: {}".format(ffn_percent))

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
    print("pruned_attn_ratio:", pruned_attn_ratio)
    print("pruned_ffn_ratio:", pruned_ffn_ratio)
    print("2 => pre-processing successful!")
    print(cfg)

    # 定义一个新的模型结构复制参数
    newmodel = ViT_slim(
        image_size=224,
        patch_size=16,
        num_classes=100,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
        qkv_bias=True,
        cfg=cfg_prune)  # 根据上面得到的剪枝cfg_prune重新定义网络
    newmodel.to(device)

    # 参数移植
    newmodel_dict = newmodel.state_dict().copy()
    i = 0
    newdict = {}
    for k, v in model.state_dict().items():
        if 'net.0.weight' in k:
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
        elif 'to_qkv' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            qkv_mask = torch.cat((cfg_mask[i], cfg_mask[i], cfg_mask[i]))
            idx = np.squeeze(np.argwhere(np.asarray(qkv_mask.cpu().numpy())))
            newdict[k] = v[idx.tolist()].clone()
        elif 'net2.0.weight' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            newdict[k] = v[:, idx.tolist()].clone()
            i = i + 1
        elif 'to_out.0.weight' in k:
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
    print("3 => load new dict successfully!")

    # 剪枝后微调
    train(newmodel, 15, attn_percent, ffn_percent)
    print("4 => pruned model: ")
    flops, params = get_model_complexity_info(newmodel, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    print('flops: ', flops, 'params: ', params)

    # 微调后测试并保存
    print("5 => after pruning: ")
    tst(newmodel)
    torch.save(newmodel.state_dict(),
               'checkpoint/pruned_pw_attn' + str(attn_percent) + '_ffn' + str(ffn_percent) + '.pth')
    filePath = 'checkpoint/pruned_pw_attn' + str(attn_percent) + '_ffn' + str(ffn_percent) + '.pth'
    print('pruned_pw_attn' + str(attn_percent) + '_ffn' + str(ffn_percent) + '.pth'+'   '+getFileSize(filePath))
    print()
    print("over!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print()





