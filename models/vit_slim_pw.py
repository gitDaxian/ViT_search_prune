import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from ptflops import get_model_complexity_info


# import ipdb

# class channel_selection(nn.Module):
#     def __init__(self, num_channels):
#         """
#         Initialize the `indexes` with all one vector with the length same as the number of channels.
#         During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
#         """
#         super(channel_selection, self).__init__()
#         self.indexes = nn.Parameter(torch.ones(num_channels))
#
#     def forward(self, input_tensor):
#         """
#         Parameter
#         ---------
#         input_tensor: (B, num_patches + 1, dim).
#         """
#         # input_tensor=[b,65,512]
#         output = input_tensor.mul(self.indexes)
#         return output

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # self.select = channel_selection(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        # x = self.select(x)
        x = self.net2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, dim1, heads=8, dim_head=64, dropout=0., qkv_bias=False):
        super().__init__()
        # inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim1 ** -0.5

        self.to_qkv = nn.Linear(dim, dim1 * 3, bias=qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(dim1, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # self.select = channel_selection(dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # q, k, v = map(lambda t: t, qkv)
        # q = self.select(q)
        # q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        # k = self.select(k)
        # k = rearrange(k, 'b n (h d) -> b h n d', h = h)
        # v = self.select(v)
        # v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, cfg, dropout=0., qkv_bias=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        if cfg is not None:
            for num in cfg:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, num[0], heads=heads, dim_head=dim_head, dropout=dropout,
                                                    qkv_bias=qkv_bias))),
                    Residual(PreNorm(dim, FeedForward(dim, num[1], dropout=dropout)))
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                                    qkv_bias=qkv_bias))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
                ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT_slim(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, cfg=None, pool='cls',
                 channels=3, dim_head=64, dropout=0., emb_dropout=0., qkv_bias=False):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            # nn.Linear(patch_dim, dim),
            # use conv2d to fit weight file
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, cfg, dropout, qkv_bias)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask=None):
        # img[b, c, img_h, img_h] > patches[b, p_h*p_w, dim]
        x = self.to_patch_embedding(img)
        x = x.flatten(2).transpose(1, 2)
        # ipdb.set_trace()
        b, n, _ = x.shape

        # cls_token[1, p_n*p_n*c] > cls_tokens[b, p_n*p_n*c]
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # add(concat) cls_token to patch_embedding
        x = torch.cat((cls_tokens, x), dim=1)
        # add pos_embedding
        x += self.pos_embedding[:, :(n + 1)]
        # drop out
        x = self.dropout(x)

        # main structure of transformer
        x = self.transformer(x, mask)

        # use cls_token to get classification message
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == "__main__":
    # setup_seed(200)
    b, c, h, w = 1, 3, 224, 224
    x = torch.randn(b, c, h, w)

    cfg_prune = [[768, 601], [768, 538], [768, 455], [768, 437], [768, 383], [768, 349], [768, 294], [768, 235],
                 [768, 168], [768, 117], [768, 88], [768, 20]]

    net = ViT_slim(
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
        cfg=cfg_prune
    )

    flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    print('flops: ', flops, 'params: ', params)

    y = net(x)

    # for m in net.modules():
    #     print(m)
    # print(net)
