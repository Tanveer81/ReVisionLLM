# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import random
from typing import Optional
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images. (To 1D sequences)
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        """
        Args:
            x: torch.tensor, (batch_size, L, d)
            mask: torch.tensor, (batch_size, L), with 1 as valid

        Returns:

        """
        assert mask is not None
        x_embed = mask.cumsum(1, dtype=x.dtype)  # (bsz, L)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=x.dtype, device=x.device)#torch.float32
        #dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t  # (bsz, L, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # (bsz, L, num_pos_feats*2)
        # import ipdb; ipdb.set_trace()
        return pos_x  # .permute(0, 2, 1)  # (bsz, num_pos_feats*2, L)


class ClipEncoder(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_encoder_layers=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, hidden_size=4096, clip_adapter_text=False,
                 cross_attn=False, hierarchy=True, clip_adapter_feature='cls'):
        super().__init__()
        if cross_attn:
            self.text_mm_projector = nn.Linear(d_model, hidden_size)
            d_model = hidden_size
        self.hidden_dim = d_model
        self.global_rep_token = torch.nn.Parameter(torch.randn(d_model))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(d_model))
        self.position_embedding = PositionEmbeddingSine(d_model, temperature=10000, normalize=True)
        self.clip_adapter_text = clip_adapter_text
        self.clip_adapter_feature = clip_adapter_feature
        self.cross_attn=cross_attn
        self.hierarchy=hierarchy

        if self.clip_adapter_text:
            t2v_encoder_layer = T2V_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.t2v_encoder = TransformerEncoder(t2v_encoder_layer, num_encoder_layers, encoder_norm)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.mm_projector = nn.Identity() if self.cross_attn else nn.Linear(d_model, hidden_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_txt=None, mask_text=None, iteration_step=None):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        if self.cross_attn:
            src_txt = self.text_mm_projector(src_txt)
        mask = torch.tensor([[1]]).to(src.device).repeat(src.shape[0], src.shape[1])
        pos = self.position_embedding(src, mask)
        mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask_, mask.bool()], dim=1)
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_, src], dim=1)
        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos_embed = torch.cat([pos_, pos], dim=1)
        pos_embed = pos_embed.permute(1, 0, 2)
        if self.clip_adapter_text:
            video_length = src.shape[0] - 1
            src_txt = src_txt.permute(1, 0, 2)
            src_t2v = torch.cat([src, src_txt], dim=0)
            pos_txt = torch.zeros_like(src_txt)  # (bsz, L_txt, d)
            pos_embed_t2v = torch.cat([pos_embed, pos_txt], dim=0)
            mask_t2v = torch.cat([mask, mask_text.bool()], dim=1)
            src_t2v = self.t2v_encoder(src_t2v, src_key_padding_mask=~mask_t2v, pos=pos_embed_t2v, video_length=video_length)
        # if self.hierarchy:
        #     src = src_t2v[:video_length + 1]
        # else:
        #     src = src_t2v[1:video_length + 1]
        #     mask = mask[:,1:video_length + 1]
        #     pos_embed = pos_embed[1:video_length + 1]
        if self.clip_adapter_text:
            src = src_t2v[:video_length + 1]
        memory, attn_weights = self.encoder(src, src_key_padding_mask=~mask, pos=pos_embed, mask=None)  # (L, batch_size, d)
        if self.clip_adapter_feature == 'alternate':
            if iteration_step%2==0: #CLS
                memory = self.mm_projector(memory[0, ...][None].permute(1, 0, 2))
            else: #Temporal
                memory = self.mm_projector(memory[1:, ...].permute(1, 0, 2))
        elif self.hierarchy or self.clip_adapter_feature=='cls':
            memory = self.mm_projector(memory[0,...][None].permute(1, 0, 2))
        elif self.clip_adapter_feature=='temporal':
            memory = self.mm_projector(memory[1:, ...].permute(1, 0, 2))
        else:
            memory = self.mm_projector(memory.permute(1, 0, 2))
        return memory


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        output = src

        intermediate = []
        attn_weights = []
        attn_weight = None
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, **kwargs)
            if isinstance(output, tuple):
                output, attn_weight = output
            if self.return_intermediate:
                intermediate.append(output)
            if attn_weight is not None:
                attn_weights.append(attn_weight)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        if attn_weight is not None:
            return output, torch.stack(attn_weights)
        else:
            return  output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
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

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, attn_weights = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights[:,0,1:]

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src , attn_weights[:,0,:]

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class T2V_TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
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

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     video_length=None):
        assert video_length is not None

        # print('before src shape :', src.shape)
        pos_src = self.with_pos_embed(src, pos)
        global_token, q, k, v = src[0].unsqueeze(0), pos_src[1:video_length + 1], pos_src[video_length + 1:], src[video_length + 1:]

        # print(src_key_padding_mask.shape) # torch.Size([32, 102])
        # print(src_key_padding_mask[:, 1:76].permute(1,0).shape) # torch.Size([75, 32])
        # print(src_key_padding_mask[:, 76:].shape) # torch.Size([32, 26])

        qmask, kmask = src_key_padding_mask[:, 1:video_length + 1].unsqueeze(2), src_key_padding_mask[:,
                                                                                 video_length + 1:].unsqueeze(1)
        attn_mask = torch.matmul(qmask.float(), kmask.float()).bool().repeat(self.nhead, 1, 1)
        # print(attn_mask.shape)
        # print(attn_mask[0][0])
        # print(q.shape) 75 32 256
        # print(k.shape) 26 32 256

        src2 = self.self_attn(q, k, value=v, attn_mask=attn_mask,
                              key_padding_mask=src_key_padding_mask[:, video_length + 1:])[0]
        src2 = src[1:video_length + 1] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[video_length + 1:]])
        # print('after src shape :',src.shape)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        print('before src shape :', src.shape)
        src2 = self.norm1(src)
        pos_src = self.with_pos_embed(src2, pos)
        global_token, q, k, v = src[0].unsqueeze(0), pos_src[1:76], pos_src[76:], src2[76:]
        # print(q.shape) # 100 32 256

        src2 = self.self_attn(q, k, value=v, attn_mask=src_key_padding_mask[:, 1:76].permute(1, 0),
                              key_padding_mask=src_key_padding_mask[:, 76:])[0]
        src2 = src[1:76] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[76:]])
        print('after src shape :', src.shape)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        # For tvsum, add kwargs
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, **kwargs)

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_w), dim=2)
    return pos


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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
