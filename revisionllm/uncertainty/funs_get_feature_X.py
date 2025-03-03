import torch
import torch.nn.functional as F


def get_average_hidden_states(
    hidden_states, layer_list, q_begin, q_end, num_dim=4096
):
    """
    Get the average hidden states of the query.
    Inputs:
    - hidden_states: the hidden_states of the query, shape: (num_hidden_layers,(batch_size, token_len, layer_dim))
    - layer_list: the list of layers to be used
    - q_begin: the beginning index of the calculated sequence
    - q_end: the ending index of the calculated sequence
    - num_dim: the unique(consistent) dimension of the hidden states
    """
    if q_begin == q_end:
        q_begin = q_end - 1

    result = torch.zeros(
        (hidden_states[0].shape[0], len(layer_list), num_dim),
        dtype=torch.float16,
    )
    for idx, layer_idx in enumerate(layer_list):
        result[:, idx, :] = torch.mean(
            hidden_states[layer_idx][:, q_begin:q_end, :], dim=1
        )
    return result


def get_last_token_hidden_states(
    hidden_states,
    layer_list,
    q_end,
    len_of_token_hidden_states_output,
    num_dim=4096,
):
    """
    Get the hidden states of the last token of the query.
    Inputs:
    - hidden_states: the hidden_states of the query, shape: (num_hidden_layers,(batch_size, token_len, hidden_size))
    - layer_list: the list of layers to be used
    - q_begin: the beginning index of the calculated sequence
    - q_end: the ending index of the calculated sequence
    - len_of_token_hidden_states_output: the number of hidden states of the last token to be output
    - num_dim: the unique(consistent) dimension of the hidden states
    """
    result = torch.zeros(
        (
            hidden_states[0].shape[0],
            len(layer_list),
            len_of_token_hidden_states_output,
            num_dim,
        ),
        dtype=torch.float16,
    )
    for idx, layer_idx in enumerate(layer_list):
        result[:, idx, :, :] = hidden_states[layer_idx][
            :, q_end - len_of_token_hidden_states_output : q_end, :
        ]
    return result


def get_prob_statistics(logits, tokens, q_begin, q_end, query=True):
    tokens = tokens.squeeze()

    probs = F.softmax(logits[:, q_begin:q_end, :], dim=2)
    probs = (
        probs.squeeze()
    )  # eliminate the dimension 0. the result: [q_end-q_begin x token_vocab_size]

    if len(probs.shape) < 2:  # if q_begin==q_end-1
        probs = probs.unsqueeze(0)

    next_token = torch.argmax(probs[-1, :])
    if probs.shape[0] > 1:
        new_probs = torch.stack(
            [
                probs[i, tokens[i + q_begin + 1]]
                for i in range(probs.shape[0] - 1)
            ],
            dim=0,
        )
        new_probs = torch.cat(
            [new_probs, probs[-1, next_token].unsqueeze(0)], dim=0
        )
    else:
        new_probs = probs[-1, next_token].unsqueeze(0)

    probs = new_probs
    probs_max = torch.max(-probs)
    probs_min = torch.min(-probs)
    probs_mean = torch.mean(-probs)
    probs_log_mean = torch.mean(-torch.log(probs + 1e-10))
    if q_end == q_begin + 1:
        probs_std = torch.zeros(probs.shape[0], dtype=torch.float16)[0]
        probs_log_std = torch.zeros(probs.shape[0], dtype=torch.float16)[0]
        # put it on the same device of probs
        probs_std = probs_std.to(probs.device)
        probs_log_std = probs_log_std.to(probs.device)
    else:
        probs_std = torch.std(-probs)
        probs_log_std = torch.std(-torch.log(probs + 1e-10))

    result = torch.stack(
        [
            probs_max,
            probs_min,
            probs_mean,
            probs_std,
            probs_log_mean,
            probs_log_std,
        ],
        dim=0,
    )

    return result


def get_entropy_statistics(logits, q_begin, q_end, query=True):
    """
    Get the entropy statistics of the output token.
    Inputs:
    - logits: the logits of the output token, shape: (batch_size, token_len, token_len)
    - q_begin: the beginning index of the calculated sequence
    - q_end: the ending index of the calculated sequence
    """
    if (not query) and q_end == q_begin:
        q_begin = q_end - 1
    probs = F.softmax(logits[:, q_begin:q_end, :], dim=2)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=2)
    entropy_max = entropy.max(dim=1).values
    entropy_min = entropy.min(dim=1).values
    entropy_mean = entropy.mean(dim=1)
    if q_end == q_begin + 1:
        entropy_std = torch.zeros(
            entropy.shape[0], dtype=torch.float16
        ).reshape(-1)
        # put it on the same device of entropy
        entropy_std = entropy_std.to(entropy.device)
    else:
        entropy_std = entropy.std(dim=1)
    result = torch.stack(
        [entropy_max, entropy_min, entropy_mean, entropy_std], dim=1
    )
    return result
