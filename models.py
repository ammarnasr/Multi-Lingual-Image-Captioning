# This file contains the model definitions for the project.

import torch
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import  AutoModelForCausalLM, GPT2Model


class MlpTransformer(nn.Module):

    def __init__(self, in_dim, h_dim, out_d = None, act=nnf.relu, dropout=0.):
        super().__init__()
        '''
        :param in_dim: input dimension
        :param h_dim: hidden dimension
        :param out_d: output dimension
        :param act: activation function
        :param dropout: dropout rate
        '''
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        '''
        :param dim_self: dimension of self attention
        :param dim_ref: dimension of reference attention
        :param num_heads: number of heads
        :param bias: whether to use bias
        :param dropout: dropout rate
        '''
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention

class TransformerLayer(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu, norm_layer = nn.LayerNorm):
        '''
        :param dim_self: dimension of self attention
        :param dim_ref: dimension of reference attention
        :param num_heads: number of heads
        :param mlp_ratio: ratio of MLP hidden dimension
        :param bias: whether to use bias
        :param dropout: dropout rate
        :param act: activation function
        :param norm_layer: normalization layer
        '''
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):

    def __init__(self, dim_self, num_heads, num_layers, dim_ref= None, mlp_ratio = 2., act=nnf.relu, norm_layer= nn.LayerNorm, enc_dec= False):
        super(Transformer, self).__init__()
        '''
        :param dim_self: dimension of self attention
        :param num_heads: number of heads
        :param num_layers: number of layers
        :param dim_ref: dimension of reference attention
        :param mlp_ratio: ratio of MLP hidden dimension
        :param act: activation function
        :param norm_layer: normalization layer
        :param enc_dec: whether to use encoder-decoder attention
        '''
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x


class TransformerMapper(nn.Module):

    def __init__(self, dim_clip, dim_embedding, prefix_length, clip_length, num_layers = 8, num_heads = 8):
        super(TransformerMapper, self).__init__()
        '''
        :param dim_clip: The size of the CLIP Visual features Embeddings
        :param dim_embedding: The size of GPT-2 Token Embeddings
        :param prefix_length: length of prefix
        :param clip_length: The number of CLIP Visual features
        :param num_layers: number of layers
        :param num_heads: number of heads
        '''
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, num_heads, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

class ClipCaptionModel(nn.Module):

    def __init__(self, prefix_length, lang , clip_length, prefix_size, num_layers):
        '''
        :param prefix_length: The Number of prefix tokens to which the CLIP features will be mapped
        :param lang: The language of the GPT2 model
        :param clip_length: The Number of CLIP Visual features
        :param prefix_size: The size of the CLIP Visual features Embeddings
        '''
        super(ClipCaptionModel, self).__init__()

        # set language and prefix length
        self.lang = lang
        self.prefix_length = prefix_length

        # load gpt2 model based on language
        if self.lang == 'english':
            # English Medium GPT2
            self.gpt = GPT2Model.from_pretrained('gpt2-medium')
        elif self.lang == 'arabic':
            # Arabic Medium GPT2
            self.gpt = AutoModelForCausalLM.from_pretrained("elgeish/gpt2-medium-arabic-poetry")

        # set embedding size and initialize The TransformerMapper to Project the CLIP features to the GPT2 embedding space
        if self.lang == 'english':
            self.gpt_embedding_size = self.gpt.wte.weight.shape[1]
        else:
            self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        
        self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length, clip_length, num_layers)

    def get_dummy_token(self, batch_size, device) :
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens, prefix, mask = None, labels = None):
        if self.lang == 'english':
            embedding_text = self.gpt.wte(tokens)
        else:
            embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

