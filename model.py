import torch
import torch.nn as nn 
import numpy as np
import math
import copy
from resnet import resnet18

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    # d_model=32, 卷积后处理的特征图维度是 7×7×32
    # nhead=8, dim_feedforward=512, dropout=0.1
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    # 添加位置编码信息
    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos
        

    def forward(self, src, pos):
                # src_mask: Optional[Tensor] = None,
                # src_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

'''
我们使用 224×224×3 的人脸图像做视线方向估计。估计的视线是一个二维向量，包含视线的偏航和俯仰。我们使用 L1 损失作为损失函数。
对于 GazeTR-Pure，我们将图像划分为 14×14 个 Patch。每个 Patch 的分辨率为16×16。我们使用 MLP 将 Patch 投影到 768 维特征向量中，并将其输入到 12 层 Transformer 中。我们将 MSA 中的头数设置为 64，将两层 MLP 的隐藏大小设置为 4096。在每个 MLP 之后使用 0.1 dropout。
对于 GazeTR-Hybrid，我们使用 ResNet-18 的卷积层进行特征提取。卷积层从人脸图像生成 7 × 7 × 512 的特征图。然后，我们使用额外的 1×1 卷积层来缩放信道，并获得 7×7×32 的特征图。我们将特征图输入到 6 层 Transformer 中。对于 Transformer ，我们将两层 MLP 的隐藏大小设置为 512，并执行 8 头自注意力机制。dropout 概率设置为 0.1。
'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 卷积层从人脸图像生成 7 × 7 × 512 的特征图。然后，我们使用额外的 1×1 卷积层来缩放信道，并获得 7×7×32 的特征图。
        maps = 32
        # 头数
        nhead = 8
        dim_feature = 7*7
        # 两层 MLP 的隐藏大小设置为 512
        dim_feedforward = 512
        dropout = 0.1
        # 6 层 Transformer
        num_layers = 6

        #  ResNet-18 卷积层进行特征提取
        self.base_model = resnet18(pretrained=False, maps=maps)

        # d_model: dim of Q, K, V 
        # nhead: seq num
        # dim_feedforward: dim of hidden linear layers
        # dropout: prob

        encoder_layer = TransformerEncoderLayer(
                  maps, 
                  nhead, 
                  dim_feedforward, 
                  dropout)

        encoder_norm = nn.LayerNorm(maps) 
        # num_encoder_layer: deeps of layers 

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        self.pos_embedding = nn.Embedding(dim_feature+1, maps)

        self.feed = nn.Linear(maps, 2)
            
        self.loss_op = nn.L1Loss()


    def forward(self, x_in):
        feature = self.base_model(x_in["face"])
        batch_size = feature.size(0)
        feature = feature.flatten(2)
        feature = feature.permute(2, 0, 1)
        
        cls = self.cls_token.repeat( (1, batch_size, 1))
        feature = torch.cat([cls, feature], 0)
        
        position = torch.from_numpy(np.arange(0, 50)).cuda()

        pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        feature = self.encoder(feature, pos_feature)
  
        feature = feature.permute(1, 2, 0)

        feature = feature[:, :, 0]

        gaze = self.feed(feature)
        
        return gaze

    def loss(self, x_in, label):
        gaze = self.forward(x_in)
        loss = self.loss_op(gaze, label) 
        return loss

