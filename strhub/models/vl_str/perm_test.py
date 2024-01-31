# coding=utf-8

import torch

def generate_attn_masks(perm):
    """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
    :param perm: the permutation sequence. i = 0 is always the BOS
    :return: lookahead attention masks
    """
    sz = perm.shape[0]
    mask = torch.zeros((sz, sz))
    for i in range(sz):
        query_idx = perm[i]
        masked_keys = perm[i + 1:]
        mask[query_idx, masked_keys] = float('-inf')
    content_mask = mask[:-1, :-1].clone()
    mask[torch.eye(sz, dtype=torch.bool)] = float('-inf')  # mask "self"
    query_mask = mask[1:, :-1]
    return content_mask, query_mask

# perms = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
#                         [0, 6, 5, 4, 3, 2, 1],
#                         [0, 3, 1, 5, 2, 4, 6],
#                         [0, 1, 4, 2, 3, 5, 6],
#                         [0, 2, 3, 5, 1, 4, 6],
#                         [0, 2, 1, 3, 5, 4, 6]])

# for perm in perms:
#     sz = perm.shape[0]
#     mask = torch.zeros((sz, sz))

#     for i in range(sz):
#         q_idx = perm[i]
#         masked_keys = perm[i + 1:]
#         mask[q_idx, masked_keys] = float('-inf')
#     print(mask)

L= 7
mask = torch.triu(torch.full((L, L), float('-inf')), 1)
content_mask = mask[:-1, :-1].clone()