# https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb

import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from performer_pytorch import SelfAttention
from linformer import LinformerSelfAttention
from reformer_pytorch import LSHSelfAttention


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_model_text=512):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model_text, d_model)
        self.W_v = nn.Linear(d_model_text, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K=None, V=None, mask=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CrossLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, d_model_text=512, self_attn=None, max_video_length=None,sa_pos=2):
        super(CrossLayer, self).__init__()
        self.cross_attn = MultiHeadAttention(d_model, num_heads, d_model_text)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.self_attn = self_attn
        self.sa_pos = sa_pos
        if self.self_attn is not None:
            self.norm1 = nn.LayerNorm(d_model)
        if self_attn == 'performer':
            self.self_attn = SelfAttention(dim = d_model, heads= num_heads, dropout=dropout)
        elif self_attn == 'linformer':
            self.self_attn = LinformerSelfAttention(dim=d_model,seq_len=max_video_length,heads=num_heads,
                k=256,one_kv_head=True, share_kv=True, dropout=dropout)
        elif self_attn == 'self-attn':
            self.self_attn = MultiHeadAttention(d_model, num_heads, d_model)

    def forward(self, video, text, text_mask):
        if self.self_attn is not None and self.sa_pos==1:
            attn_output = self.self_attn(video)
            video = self.norm1(video + self.dropout(attn_output))
        attn_output = self.cross_attn(video, text, text, text_mask)
        video = self.norm2(video + self.dropout(attn_output))
        if self.self_attn is not None and self.sa_pos==2:
            attn_output = self.self_attn(video)
            video = self.norm1(video + self.dropout(attn_output))
        ff_output = self.feed_forward(video)
        video = self.norm3(video + self.dropout(ff_output))
        return video


class PerformerSelfLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(PerformerSelfLayer, self).__init__()
        self.self_attn = SelfAttention(dim = d_model, heads= num_heads, dropout=dropout, dim_head = d_model//num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, video):
        attn_output = self.self_attn(video, video, video, None)
        video = self.norm2(video + self.dropout(attn_output))
        ff_output = self.feed_forward(video)
        video = self.norm3(video + self.dropout(ff_output))
        return video

class CrossAttn(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_video_length, dropout, text_dim=512,
                 max_text_length=100, out=None, self_attn=None, ca_self_attn=None, sa_pos=2):
        super(CrossAttn, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_video_length)
        self.text_positional_encoding = PositionalEncoding(text_dim, max_text_length)
        self.cross_layers = nn.ModuleList([CrossLayer(d_model, num_heads, d_ff, dropout, text_dim, ca_self_attn, max_video_length,sa_pos) for _ in range(num_layers)])
        self.self_attn = self_attn
        if self.self_attn == 'performer':
            self.self_layers = nn.ModuleList([PerformerSelfLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.out_layer = nn.Linear(d_model, out) if out is not None else None
        self.dropout = nn.Dropout(dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, video, text, text_mask=None):
        video = self.dropout(self.positional_encoding(video))
        text = self.dropout(self.text_positional_encoding(text))
        for cross_layer in self.cross_layers:
            video = cross_layer(video, text, text_mask.unsqueeze(1).unsqueeze(2).bool())
        if self.self_attn is not None:
            for self_layer in self.self_layers:
                video = self_layer(video)
        if self.out_layer is not None:
            video = self.out_layer(video)
        return video


if __name__ == "__main__":
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    crossattn = CrossAttn(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

    # Generate random sample data
    video_feat = torch.randint(1, src_vocab_size, (64, 100))  # (batch_size, seq_length)
    text_feat = torch.randint(1, tgt_vocab_size, (64, 8))  # (batch_size, seq_length)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(crossattn.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    crossattn.train()

    for epoch in range(100):
        optimizer.zero_grad()
        output = crossattn(text_feat, video_feat[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), text_feat[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")