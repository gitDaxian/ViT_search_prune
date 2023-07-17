import torch
import torchvision
from models import VGG
from models.vit import ViT, channel_selection
from models.vit_slim import ViT_slim
from ptflops import get_model_complexity_info

vit = ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 512,                  # 512
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
    )
vgg = VGG('VGG19')


# print('vit model: ')
# flops, params = get_model_complexity_info(vit, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
# print('flops: ', flops, 'params: ', params)
#
# print('vgg model: ')
# flops2, params2 = get_model_complexity_info(vgg, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
# print('flops: ', flops2, 'params: ', params2)

import json

if __name__ == '__main__':
    fjson = "checkpoint/model_cfg_prune.json"
    # 读取原始json文件
    with open(fjson, 'r') as f:
        content = json.load(f)
    # print(content["imagenet_pruned_attn0_ffn0.pth"])
    # 更新字典dict
    axis = {"imagenet_pruned_attn0_ffn0.pth":
                [768, 3072, 768, 3072, 768, 3072, 768, 3072, 768, 3072, 768, 3072, 768, 3072, 768, 3072,
                 768, 3072, 768, 3072, 768, 3072, 768, 3072]}
    content.update(axis)
    # 写入
    with open(fjson, 'w') as f_new:
        json.dump(content, f_new)

    # 截断保留n位小数
    # ffn_ratio = 0.88
    # ffn_ratio2 = 0.1
    #
    # ffn_str = str(ffn_ratio2 - 0.1) if len(str(ffn_ratio2).split(".")[1]) == 1 else str(int(ffn_ratio2 * 10) / 10)
    # print(ffn_str)
    # print(int(ffn_ratio*10)/10)
    #
    # print(len(str(ffn_ratio).split(".")[1]))
    # print(len(str(ffn_ratio2).split(".")[1]))

    # ffn_total = 7373
    # ffn_ratio = 0.85
    # real_ffn_ratio = (ffn_total - 36864 * (1 - ffn_ratio)) / ffn_total
    # print(real_ffn_ratio)
    # print(type(real_ffn_ratio))

    pass






