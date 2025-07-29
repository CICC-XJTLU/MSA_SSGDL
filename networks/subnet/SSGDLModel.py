import torch
import torch.nn as nn
import torch.nn.functional as F
from config.global_configs import *
import math
import time
import numpy as np
from collections import OrderedDict

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class SKAttention(nn.Module):
    def __init__(self, in_channel=768, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super(CSDSA, self).__init__()
        self.in_channels = in_channel
        self.channels = in_channel // 8
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        self.SoftMax = nn.Softmax(dim=3)
        self.INF = INF 
        self.gamma = nn.Parameter(torch.zeros(1))
        self.d = max(L, in_channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, in_channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(in_channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(in_channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, in_channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        b, c, h, w = x.size()
        conv_outs = []
        for conv in self.convs:
            query = self.ConvQuery(x)
            query_H = query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
            query_W = query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)

            key = self.ConvKey(x)
            key_H = key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
            key_W = key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

            value = self.ConvValue(x)
            value_H = value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
            value_W = value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

            energy_H = (torch.bmm(query_H, key_H) + self.INF(b, h, w)).view(b, w, h, h).permute(0, 2, 1, 3)
            energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)

            concate = self.SoftMax(torch.cat([energy_H, energy_W], 3))

            attention_H = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
            attention_W = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w)

            out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
            out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)

            x = out_H + out_W
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)

        U = sum(conv_outs)

        S = U.mean(-1).mean(-1)  
        Z = self.fc(S) 

        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(b, c, 1, 1))  
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)  

        V = (attention_weights * feats).sum(0)
        return V


class CSDSA(nn.Module):
    def __init__(self, in_channels):
        super(CSDSA, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 8
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        self.SoftMax = nn.Softmax(dim=3)
        self.INF = INF 
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        

        
        return self.gamma * () + x


class SSGDL(nn.Module):
    def __init__(self, beta_shift_a=0.5, beta_shift_v=0.5, dropout_prob=0.2, name=""):
        super(SSGDL, self).__init__()
        self.visual_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.acoustic_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.sa = SelfAttention(TEXT_DIM)
        self.cat_connect = nn.Linear(2 * TEXT_DIM, TEXT_DIM)
        self.cat_connect_fusion = nn.Linear(3 * TEXT_DIM, TEXT_DIM)
        self.cross_ATT_visual = CrossAttention(dim=TEXT_DIM)
        self.cross_ATT_acoustic = CrossAttention(dim=TEXT_DIM)
        self.cross_ATT_textual = CrossAttention(dim=TEXT_DIM)
        self.layer_norm = nn.LayerNorm(TEXT_DIM)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(TEXT_DIM, TEXT_DIM, batch_first=True)  # 使用LSTM替代GRU
        self.ssgdl = SSGDL(TEXT_DIM)
        
    def forward(self, text_embedding, visual=None, acoustic=None, visual_ids=None, acoustic_ids=None):
        # Call external memory module
        external_memory_output = self.memory_module(text_embedding)
        visual_ = self.visual_embedding(visual_ids)
        acoustic_ = self.acoustic_embedding(acoustic_ids) 
        # 1. Embedding layer, The parameters of the embedding layer are learnable, map the nonverbal index vector,
        # which is obtained by the feature transformation strategy, to a high-dimensional space.
        visual_ = self.cross_ATT_visual(text_embedding, visual_, visual_)
        acoustic_ = self.cross_ATT_acoustic(text_embedding, acoustic_, acoustic_)

        # textual
        b, n, c = text_embedding.size()
        text_embedding_re = text_embedding.view(b, c, h, w)
        text_embedding = self.ssgdl(text_embedding_re)
        b, c, h, w = text_embedding.size()
        text_embedding = text_embedding.view(b, n, c)

        #visual
        visual_ = torch.relu(visual_)
        h, w = 1, 50
        b, n, c = visual_.size()
        visual_re = visual_.view(b, c, 1, n)  # 调整visual_的形状，注意参数顺序
        visual_ = self.ssgdl(visual_re)
        b, c, h, w = visual_.size()
        visual_ = visual_.view(b, n, c)
        visual_lstm, _ = self.lstm(visual_)
    
        # acoustic
        acoustic_lstm, _ = self.lstm(acoustic_)
        b, n, c = acoustic_.size()
        acoustic_re = acoustic_.view(b, c, h, w)
        acoustic_ = self.ssgdl(acoustic_re)
        b, c, h, w = acoustic_.size()
        acoustic_ = acoustic_.view(b, n, c)

        # fusion
        fusion = torch.cat((visual_, acoustic_, text_embedding), dim=-1)
        fusion = self.cat_connect_fusion(fusion)
        b, n, c = fusion.size()
        fusion_re = fusion.view(b, c, h, w)
        fusion = self.ssgdl(fusion_re)
        b, c, h, w = fusion.size()
        fusion = fusion.view(b, n, c)

        # [a->v]_[t->v]
        visual_a = self.cross_ATT_visual(visual_, acoustic_, acoustic_)
        visual_t = self.cross_ATT_textual(visual_, text_embedding, text_embedding)
        visual_ = torch.cat(visual_a, visual_t)
        visual_ = self.cat_connect(visual_)

        # [a->t]_[v->t]
        text_embedding_a = self.cross_ATT_visual(text_embedding, acoustic_, acoustic_)
        text_embedding_v = self.cross_ATT_textual(text_embedding, visual_, visual_)
        text_embedding = torch.cat(text_embedding_a, text_embedding_v)
        text_embedding = self.cat_connect(text_embedding)

        # [v->a]_[t->a]
        acoustic_v = self.cross_ATT_visual(acoustic_, visual_, visual_)
        acoustic_t = self.cross_ATT_textual(acoustic_, text_embedding, text_embedding)
        acoustic_ = torch.cat(acoustic_t, acoustic_v)
        acoustic_ = self.cat_connect(acoustic_)

        # [a->f]_[t->f]_[v->f]
        acoustic_f = self.cross_ATT_textual(fusion, acoustic_, acoustic_)
        text_embedding_f = self.cross_ATT_textual(fusion, text_embedding, text_embedding)
        visual_f = self.cross_ATT_textual(fusion, visual_, visual_)
        fusion = torch.cat(acoustic_f, visual_f, text_embedding_f)
        fusion = self.cat_connect(fusion)
        fusion = self.sa(fusion)

        final_fusion = 0.6 * text_embedding + 0.3 * visual_ + 0.3 * acoustic_ + fusion

        shift = self.layer_norm(final_fusion)  # Residual connection and layer normalization
        shift = self.dropout(shift)  # Apply dropout

        # Apply LSTM
        output, _ = self.lstm(shift)

        # Final output
        embedding_shift = output + text_embedding

        return embedding_shift



class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_per_head = dim // heads
        self.scale = self.dim_per_head ** -0.5

        assert dim % heads == 0, "Dimension must be divisible by the number of heads."

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, queries, keys, values, mask=None):
        
        b, n, d = queries.size()
        h = self.heads

        queries = self.query(queries).view(b, n, h, self.dim_per_head).transpose(1, 2)
        keys = self.key(keys).view(b, -1, h, self.dim_per_head).transpose(1, 2)  # Fix here
        values = self.value(values).view(b, -1, h, self.dim_per_head).transpose(1, 2)  # Fix here

        dots = torch.einsum('bhqd,bhkd->bhqk', queries, keys) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, h, -1)
            dots.masked_fill_(~mask, float('-inf'))

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhqk,bhvd->bhqd', attn, values)
        out = out.transpose(1, 2).contiguous().view(b, n, d)

        return out

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, head_num=1):
        super(SelfAttention, self).__init__()
        self.head_num = head_num
        self.s_d = hidden_size // self.head_num
        self.all_head_size = self.head_num * self.s_d
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        x = x.view(x.size(0), x.size(1), self.head_num, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, text_embedding, embedding):
        Q = self.Wq(text_embedding)
        K = self.Wk(embedding)
        V = self.Wv(embedding)
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)
        weight_score = torch.matmul(Q, K.transpose(-1, -2))
        weight_prob = nn.Softmax(dim=-1)(weight_score * 8)

        context_layer = torch.matmul(weight_prob, V)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
