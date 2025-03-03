import torch
import torch.nn.functional as F


def span_cxw_to_xx(cxw_spans):
    """
    Args:
        cxw_spans: tensor, (#windows, 2) or (..., 2), the last dim is a row denoting a window of format (center, width)

    >>> spans = torch.Tensor([[0.5000, 1.0000], [0.3000, 0.2000]])
    >>> span_cxw_to_xx(spans)
    tensor([[0.0000, 1.0000],
        [0.2000, 0.4000]])
    >>> spans = torch.Tensor([[[0.5000, 1.0000], [0.3000, 0.2000]]])
    >>> span_cxw_to_xx(spans)
    tensor([[[0.0000, 1.0000],
        [0.2000, 0.4000]]])
    """
    x1 = cxw_spans[..., 0] - 0.5 * cxw_spans[..., 1]
    x2 = cxw_spans[..., 0] + 0.5 * cxw_spans[..., 1]
    return torch.stack([x1, x2], dim=-1)


def forward_clip_matching( src_cls_txt, src_vid_appear, src_vid_appear_mask, proposal=None,
                          is_groundtruth=False):
    """
    The forward expects following tensors:
        - src_cls_txt: [batch_size, D_txt]
        - src_vid_appear: [batch_size, L_vid, D_vid]
        - src_vid_appear_mask: [batch_size, L_vid], containing 0 on padded pixels
        - proposal:
        - is_groundtruth: whether the proposal comes from the ground-truth (during training)
        or proposal generation prediction (during inference).
    It returns a proposal-query similarity matrix.
    """
    text_cls_features = src_cls_txt / src_cls_txt.norm(dim=1, keepdim=True)

    proposal_score = _get_predicted_proposal_feat(src_vid_appear, src_vid_appear_mask, proposal, text_cls_features)
    #proposal_features = proposal_feat / proposal_feat.norm(dim=2, keepdim=True)
    #return torch.einsum('bld,bd->bl', proposal_features, text_cls_features)
    return proposal_score


def _get_predicted_proposal_feat( src_vid_appear, src_vid_appear_mask, pred_proposal, text_cls_features):
    """
    The forward expects following tensors:
      - src_vid_appear: [batch_size, L_vid, D_vid]
      - src_vid_appear_mask: [batch_size, L_vid], containing 0 on padded pixels
      - proposal: [batch_size, N_query, 2], predicted start and end timestamps for each moment queries
    It returns proposal features for predicted proposals.
    """
    duration = torch.sum(src_vid_appear_mask, dim=-1)
    proposal = torch.einsum('bld,b->bld', span_cxw_to_xx(pred_proposal), duration)  # .to(torch.int32)
    bsz, n_query = proposal.shape[:2]
    proposal_start = F.relu(torch.floor(proposal[:, :, 0]).to(torch.int32))
    proposal_end = torch.ceil(proposal[:, :, 1]).to(torch.int32)
    proposal_feat_list = []
    for idx, (feat, start_list, end_list) in enumerate(zip(src_vid_appear, proposal_start, proposal_end)):
        for start, end in zip(start_list, end_list):
            clip_feat = feat[start:end]
            clip_feat = clip_feat / clip_feat.norm(dim=1, keepdim=True)
            clip_feat = _topk_pooling(text_cls_features[idx][None], clip_feat[None], min(clip_feat.shape[0], 3))[0]
            #clip_feat = _attention_pooling(text_cls_features[idx][None], clip_feat[None], 0.01)[0]
            score = torch.einsum('ld,d->l', clip_feat, text_cls_features[idx])
            #proposal_feat_list.append(torch.topk(score, min(clip_feat.shape[0], 3), 0)[0].mean(0))
            proposal_feat_list.append(score)
    proposal_feat = torch.vstack(proposal_feat_list)
    proposal_feat = proposal_feat.reshape(bsz, n_query)#, vid_appear_dim)
    return proposal_feat

def _topk_pooling( text_embeds, video_embeds, k):
    """
    Pooling top-k frames for each video based on
    similarities with each text query

    Output
        video_embeds_pooled: num_vids x num_texts x embed_dim
    """
    num_texts, embed_dim = text_embeds.shape

    # num_vids x num_frames x num_texts
    sims = video_embeds @ text_embeds.t()
    sims_topk = torch.topk(sims, k, dim=1)[1]

    # Make format compatible with torch.gather
    video_embeds = video_embeds.unsqueeze(-1).expand(-1, -1, -1, num_texts)
    sims_topk = sims_topk.unsqueeze(2).expand(-1, -1, embed_dim, -1)

    # num_vids x k x embed_dim x num_texts
    video_embeds_topk = torch.gather(video_embeds, dim=1, index=sims_topk)

    # Top-k pooling => num_vids x embed_dim x num_texts
    video_embeds_pooled = video_embeds_topk.sum(dim=1)
    return video_embeds_pooled.permute(0, 2, 1)

def _attention_pooling( text_embeds, video_embeds, temperature):
    """
    Pooling frames for each video using attention-based
    similarity with each text query

    Output
        video_embeds_pooled: num_vids x num_texts x embed_dim
    """
    # num_vids x num_frames x num_texts
    sims = video_embeds @ text_embeds.t()
    attention_weights = F.softmax(sims / temperature, dim=1)

    # num_vids x embed_dim x num_frames
    video_embeds = video_embeds.permute(0, 2, 1)

    # num_vids x embed_dim x num_texts
    video_embeds_pooled = torch.bmm(video_embeds, attention_weights)
    return video_embeds_pooled.permute(0, 2, 1)
