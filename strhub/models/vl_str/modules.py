# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import math
from functools import partial
from typing import Optional, Sequence, Callable

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import transformer
from timm.models.helpers import named_apply
from strhub.models.utils import init_weights


class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
       This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward_stream(self, tgt: Tensor, tgt_norm: Tensor, tgt_kv: Tensor, memory: Tensor, attn_mask: Optional[Tensor],
                       key_padding_mask: Optional[Tensor]):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        tgt2, sa_weights = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(self, query, content, memory, tgt_query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None, update_content: bool = True):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, tgt_query_mask, content_key_padding_mask)[0]
        if update_content:
            content = self.forward_stream(content, content_norm, content_norm, memory, content_mask, content_key_padding_mask)[0]
        return query, content


class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm, embed_dim=512, dropout=0.0, num_classes=94,
                        charset_size=94, max_label_length=25):
        """a self-contained decoder for character extraction"""
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.text_embed = TokenEmbedding(charset_size, embed_dim)
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length, embed_dim))
        self.num_layers = num_layers
        self.norm = norm
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(embed_dim, num_classes, bias=True)

        named_apply(partial(init_weights, exclude=['none']), self)
        nn.init.trunc_normal_(self.pos_queries, std=.02)

    def forward(self, tgt, memory,
                tgt_query: Optional[Tensor] = None,
                tgt_query_mask: Optional[Tensor] = None,
                content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        content = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        query = self.dropout(tgt_query)

        # forward layers
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(query, content, memory, tgt_query_mask, content_mask, content_key_padding_mask,
                                 update_content=not last)
        query = self.norm(query)

        # prediction
        logits = self.head(query)

        # return prediction and feature
        return logits, query


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class Hook():
    # A simple hook class that returns the input and output of a layer during forward/backward pass
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def modify_attn_mask(m, context_length=10):
    if hasattr(m, "attn_mask"):
        if m.attn_mask is not None:
            m.attn_mask = m.attn_mask[:context_length, :context_length]


class FusionFC(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, num_classes)
        # init
        for m in self.modules():
            init_weights(m)

    def forward(self, feature1, feature2, detach=True):
        """
        Args:
            feature1: (N, T, E) where T is length, N is batch size and d is dim of model
            feature2: (N, T, E) shape the same as l_feature 
        """
        if detach:
            feature1 = feature1.detach().clone()
            feature2 = feature2.detach().clone()
        f = torch.cat((feature1, feature2), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * feature2 + (1.0 - f_att) * feature1
        # (N, T, C)
        logits = self.cls(output)

        return logits
