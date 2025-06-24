import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from utils import numberClassChannel
from ikan.GroupKAN import GroupKAN, GroupKANLinear

class MPE_CNN(nn.Module):
    def __init__(self, f1=8, kernel_sizes=(32, 64, 96), num_groups=8,D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3,
                 number_channel=22, emb_size=16):
        """
        PatchEmbeddingCNN的初始化函数

        :param f1: 第一个卷积层的输出通道数，默认为16
        :param kernel_sizes: 不同卷积核大小的元组，默认为(32, 64, 96)
        :param D: 深度可分离卷积的倍数，默认为2
        :param pooling_size1: 第一次平均池化的窗口大小，默认为8
        :param pooling_size2: 第二次平均池化的窗口大小，默认为8
        :param dropout_rate: Dropout的概率，默认为0.3
        :param number_channel: 输入数据的通道数，默认为3
        :param emb_size: 嵌入的特征维度大小，默认为16
        """
        super().__init__()
        f2 = D * f1
        print("--------------------------D", D)
        print("--------------------------f1", f1)
        print("--------------------------f2", f2)
        self.conv_layers = nn.ModuleList()  # 使用ModuleList来存放不同kernel_size的卷积模块
        print("k_s:",kernel_sizes,num_groups)
        # 构建多个不同kernel_size的卷积模块
        for kernel_size in kernel_sizes:
            conv_module = nn.Sequential(
                # temporal conv
                nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False),
                nn.BatchNorm2d(f1),
                # channel depth-wise conv
                nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                # average pooling 1
                nn.AvgPool2d((1, pooling_size1)),
                nn.Dropout(dropout_rate),
                # spatial conv
                nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),

                # average pooling 2 to adjust the length of feature into transformer encoder
                nn.AvgPool2d((1, pooling_size2)),
                nn.Dropout(dropout_rate),

            )
            self.conv_layers.append(conv_module)

        # 额外添加的卷积层，用于对三个张量进行卷积合并
        self.combine_conv = nn.Conv2d(f2 * len(kernel_sizes), f2, (1, 1), bias=False)

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数

        :param x: 输入的张量数据
        :return: 经过处理后的张量数据
        """
        outputs = []
        for conv_layer in self.conv_layers:
            out = conv_layer(x)
            outputs.append(out)

        # 在通道维度上拼接三个张量
        # print("After conv2:", x.shape)
        combined = torch.cat(outputs, dim=1)
        # print("After conv3:", combined.shape)

        # 通过额外的卷积层进行卷积合并
        combined = self.combine_conv(combined)
        # print("After conv4:", combined.shape)

        x = self.projection(combined)
        # print("After conv5:", x.shape)

        return x

# BranchEEGNetTransformer类，包含PatchEmbeddingCNN模块
class MCTG(nn.Sequential):
    def __init__(self, heads=4,
                 depth=6,
                 emb_size=16,
                 number_channel=22,
                 f1=8,
                 kernel_sizes=(32, 64, 128),
                 D=2,
                 pooling_size1=8,
                 pooling_size2=8,
                 dropout_rate=0.3,
                 num_groups=8,
                 **kwargs):
        """
        BranchEEGNetTransformer的初始化函数

        :param heads: 多头注意力的头数量，默认为4
        :param depth: 编码器块的深度（数量），默认为6
        :param emb_size: 嵌入的特征维度大小，默认为40
        :param number_channel: 输入数据的通道数，默认为22
        :param f1: 第一个卷积层的输出通道数，默认为20
        :param kernel_sizes: 不同卷积核大小的元组，默认为(32, 64, 128)
        :param D: 深度可分离卷积的倍数，默认为2
        :param pooling_size1: 第一次平均池化的窗口大小，默认为8
        :param pooling_size2: 第二次平均池化的窗口大小，默认为8
        :param dropout_rate: Dropout的概率，默认为0.3
        """
        super().__init__(
            MPE_CNN(f1=f1,
                              kernel_sizes=kernel_sizes,
                              D=D,
                              num_groups=num_groups,
                              pooling_size1=pooling_size1,
                              pooling_size2=pooling_size2,
                              dropout_rate=dropout_rate,
                              number_channel=number_channel,
                              emb_size=emb_size),
        )

class TransformerModule(nn.Module):
    """
    Stacks multiple transformer encoder layers.
    Args:
        embed_dim: Dimensionality of input embeddings.
        num_heads: Number of attention heads in each encoder layer.
        fc_ratio: Expansion ratio for feed-forward layers.
        depth: Number of transformer encoder layers.
        attn_drop: Dropout rate for attention mechanism.
        fc_drop: Dropout rate for feed-forward layers.
    """
    def __init__(self, embed_dim, num_heads, fc_ratio, depth, attn_drop, fc_drop,num_groups=None):
        super().__init__()
        # Create a list of transformer encoder layers
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop,num_groups=num_groups) for _ in range(depth)
        ])

    def forward(self, x):
        """
        Forward pass for the transformer module.
        Args:
            x: Input tensor of shape (batch_size, embed_dim, num_samples).
        Returns:
            Transformed tensor with the same shape.
        """
        # x = rearrange(x, 'b d n -> b n d')  # Rearrange to (batch_size, seq_len, embed_dim)
        for encoder in self.transformer_encoders:
            x = encoder(x)  # Pass through each encoder layer
        # x = x.transpose(1, 2)  # Rearrange back to (batch_size, embed_dim, seq_len)
        # x = x.unsqueeze(dim=2)  # Add a spatial dimension -> (batch_size, embed_dim, 1, seq_len)
        return x

# Transformer Encoder Layer
class TransformerEncoder(nn.Module):
    """
    A single transformer encoder layer with multi-head attention and feed-forward network.
    Args:
        embed_dim: Dimensionality of input embeddings.
        num_heads: Number of attention heads.
        fc_ratio: Ratio for expanding the feed-forward hidden layer.
        attn_drop: Dropout rate for attention mechanism.
        fc_drop: Dropout rate for feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5,num_groups=8):
        super().__init__()
        self.multihead_attention = MultiHeadedAttention(embed_dim, num_heads, attn_drop,num_groups=num_groups)
        self.feed_forward = FeedForward(embed_dim, embed_dim * fc_ratio, fc_drop,num_groups=num_groups)
        self.layernorm1 = nn.LayerNorm(embed_dim)  # LayerNorm after attention
        self.layernorm2 = nn.LayerNorm(embed_dim)  # LayerNorm after feed-forward

    def forward(self, data):
        # Apply attention with residual connection and layer normalization
        res = self.layernorm1(data)
        out = data + self.multihead_attention(res, res, res)

        # Apply feed-forward network with residual connection and layer normalization
        res = self.layernorm2(out)
        output = out + self.feed_forward(res)
        return output

class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention mechanism combining local and global attention.
    Args:
        d_model: Dimensionality of input features.
        n_head: Number of attention heads.
        dropout: Dropout rate for regularization.
    """
    def __init__(self, d_model, n_head, dropout,num_groups=8):
        super().__init__()
        self.d_k = d_model // n_head  # Dimensionality per attention head for keys
        self.d_v = d_model // n_head  # Dimensionality per attention head for values
        self.n_head = n_head

        # Multi-scale convolution settings for local feature extraction
        kernel_sizes = [3, 5]
        padding = [1, 2]

        self.multi_scale_conv_k = MultiScaleConv1d(d_model, d_model, kernel_sizes, padding)

        # Linear projections for queries, local keys, global keys, and values
        self.w_q = GroupKANLinear(d_model, n_head * self.d_k,num_groups=num_groups)
        self.w_k_local = GroupKANLinear(d_model * len(kernel_sizes), n_head * self.d_k,num_groups=num_groups)
        self.w_k_global = GroupKANLinear(d_model, n_head * self.d_k,num_groups=num_groups)
        self.w_v = GroupKANLinear(d_model, n_head * self.d_v,num_groups=num_groups)
        self.w_o = GroupKANLinear(n_head * self.d_v, d_model,num_groups=num_groups)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Forward pass for local and global attention combination.
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model).
            key: Key tensor of shape (batch_size, seq_len, d_model).
            value: Value tensor of shape (batch_size, seq_len, d_model).
        """
        bsz = query.size(0)

        # Local key extraction using multi-scale convolution
        key_local = key.transpose(1, 2)  # Transpose to (batch_size, d_model, seq_len)
        key_local = self.multi_scale_conv_k(key_local).transpose(1, 2)

        # Linear projections
        q = self.w_q(query).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)  # Query
        k_local = self.w_k_local(key_local).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)  # Local Key
        k_global = self.w_k_global(key).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)  # Global Key
        v = self.w_v(value).view(bsz, -1, self.n_head, self.d_v).transpose(1, 2)  # Value

        # Local attention
        scores_local = torch.matmul(q, k_local.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_local = F.softmax(scores_local, dim=-1)
        attn_local = self.dropout(attn_local)
        x_local = torch.matmul(attn_local, v)

        # Global attention
        scores_global = torch.matmul(q, k_global.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_global = F.softmax(scores_global, dim=-1)
        attn_global = self.dropout(attn_global)
        x_global = torch.matmul(attn_global, v)

        # Combine local and global attention outputs
        x = x_local + x_global

        # Concatenate results and project to output dimensions
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.n_head * self.d_v)
        return self.w_o(x)


# Feed-Forward Neural Network
class FeedForward(nn.Module):
    """
    Two-layer feed-forward network with GELU activation.
    Args:
        d_model: Dimensionality of input and output features.
        d_hidden: Dimensionality of the hidden layer.
        dropout: Dropout rate for regularization.
    """
    def __init__(self, d_model, d_hidden, dropout,num_groups=8):
        super().__init__()
        self.w_1 = GroupKANLinear(d_model, d_hidden,num_groups=num_groups)
        # self.act = nn.GELU()  # Activation function
        self.w_2 = GroupKANLinear(d_hidden, d_model,num_groups=num_groups)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)  # Linear layer 1
        # x = self.act(x)  # Activation
        x = self.dropout(x)  # Dropout
        x = self.w_2(x)  # Linear layer 2
        x = self.dropout(x)  # Dropout
        return x

class MultiScaleConv1d(nn.Module):
    """
    Multi-scale 1D convolution module to extract features with multiple kernel sizes.
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels for each convolution.
        kernel_sizes: List of kernel sizes for each convolution layer.
        padding: List of padding values for each kernel size.
    """
    def __init__(self, in_channels, out_channels, kernel_sizes, padding):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=p) for k, p in zip(kernel_sizes, padding)
        ])
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes))  # Batch normalization after concatenation
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        # Apply each convolution and concatenate the results along the channel dimension
        conv_outs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)  # Concatenate along channel axis
        out = self.bn(out)  # Apply batch normalization
        out = self.dropout(out)  # Apply dropout
        return out

class MCTGNet(nn.Module):
    def __init__(self, num_groups=8,heads=4,
                 emb_size=16,
                 depth=6,
                 database_type='A',
                 eeg1_f1=20,
                 eeg1_kernel_sizes=(16,64,128),
                 eeg1_D=2,
                 eeg1_pooling_size1=8,
                 eeg1_pooling_size2=8,
                 eeg1_dropout_rate=0.3,
                 eeg1_number_channel=22,
                 flatten_eeg1=240,
                 **kwargs):
        """
        EEGTransformer的初始化函数

        :param heads: 多头注意力的头数量，默认为4
        :param emb_size: 嵌入的特征维度大小，默认为40
        :param depth: 编码器块的深度（数量），默认为6
        :param database_type: 数据库类型，默认为'B'
        :param eeg1_f1: EEG1部分的第一个卷积层的输出通道数，默认为20
        :param eeg1_kernel_sizes: EEG1部分不同卷积核大小的元组，默认为(32, 64, 128)
        :param eeg1_D: EEG1部分深度可分离卷积的倍数，默认为2
        :param eeg1_pooling_size1: EEG1部分第一次平均池化的窗口大小，默认为8
        :param eeg1_pooling_size2: EEG1部分第二次平均池化的窗口大小，默认为8
        :param eeg1_dropout_rate: EEG1部分Dropout的概率，默认为0.3
        :param eeg1_number_channel: EEG1部分输入数据的通道数，默认为3
        :param flatten_eeg1: 展平后的特征数量，默认为600
        """
        super().__init__()
        self.number_class, self.number_channel = numberClassChannel(database_type)
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.num_groups=num_groups
        self.flatten = nn.Flatten()
        self.cnn = MCTG(heads, depth, emb_size, number_channel=self.number_channel,
                                           f1=eeg1_f1,
                                           kernel_sizes=eeg1_kernel_sizes,
                                           D=eeg1_D,
                                           pooling_size1=eeg1_pooling_size1,
                                           pooling_size2=eeg1_pooling_size2,
                                           dropout_rate=eeg1_dropout_rate,
                                           num_groups=self.num_groups
                                           )
        # self.position = PositioinalEncoding(emb_size, dropout=0.1)
        # self.trans = TransformerEncoderctnet(heads, depth, emb_size)
        self.trans = TransformerModule(16, heads, 2, depth, 0.5, 0.5,num_groups=self.num_groups)

        self.flatten = nn.Flatten()
        # self.classification = ClassificationHead(self.flatten_eeg1, self.number_class)  # FLATTEN_EEGNet + FLATTEN_cnn_module
        self.groupkan = GroupKAN(
                                layers_hidden = [240,self.number_class],
                                act_mode="gelu", 
                                drop=0.5,
                                num_groups=self.num_groups  # 添加组数参数
                )
        # 实例化 Dropout 层
        # self.dropout = nn.Dropout(0.5)
        # self.group = GroupKANLinear(self.flatten_eeg1, self.number_class)
        # self.group = nn.Linear(self.flatten_eeg1, self.number_class)

    def forward(self, x):
        """
        前向传播函数

        :param x: 输入的张量数据
        :return: 模型输出的特征和分类结果的张量数据
        """
        org = x
        cnn = self.cnn(x)
        trans = self.trans(cnn)
        # residual connect
        features = cnn + trans
        x = self.flatten(features)
        out = self.groupkan(x)
        return org,cnn,features,out