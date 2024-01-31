# coding = utf-8
import torch
from strhub.clip import clip


@torch.no_grad()
def clip_post_process(clip_model, image, probs, charset_adapter, char_tokenizer,
                        K=10, K2=5, num_samples=50, prompt=None, alpha=0.3):
    """using CLIP to do post-refinement
    Args:
        clip_model: the clip model
        image: input image, by default, after transforms
        probs: [L, C], probability distribution
        K: the sampled captions
        K2: if K2 > 0, before sampling, choose the top-K2 probabilities
        charset_adapter: charset adapter
    """
    max_p_y, max_ids = probs.max(-1)
    max_p_y_pord = max_p_y.prod(dim=0)
    p_y_prod, p_y, p_y_ind = top_k_candidate_v2(probs, K=K, num_samples=num_samples, K2=K2)

    # if the prediction with max probability not sampled, mannully add it
    if p_y_prod.max() < max_p_y_pord:
        try:
            p_y_prod = torch.cat([p_y_prod, max_p_y_pord.unsqueeze(0)], dim=0)
            p_y = torch.cat([p_y, max_p_y.unsqueeze(0)], dim=0)
            p_y_ind = torch.cat([p_y_ind, max_ids.unsqueeze(0)], dim=0)
        except:
            print(p_y.shape, p_y_ind.shape)

    text, preds = [], []
    for ind in p_y_ind:
        ids = ind.tolist()
        # maybe a single id
        ids = ids if isinstance(ids, list) else [ids]
        tokens = char_tokenizer._ids2tok(ids, join=True)

        # charset_adapter is necessary
        string = charset_adapter(tokens)
        # add prompt or not
        if prompt is not None:
            text.append(prompt + " " + string)
        else:
            text.append(string)
        preds.append(string)

    if len(text) > 1:
        with torch.no_grad():
            text_input = clip.tokenize(text).to(image.device)
            logits_from_image, logits_from_text = clip_model(image, text_input)
            clip_prob = logits_from_image.softmax(dim=-1)
    else:
        clip_prob = p_y_prod.softmax(-1)

    # combine the probabilities
    final_prob = p_y_prod.softmax(-1) + alpha * clip_prob
    p, ind = final_prob.max(dim=-1)

    final_p_y, final_ind, final_pred = p_y[ind, ...], p_y_ind[ind, ...], preds[ind.item()]

    return final_p_y, final_ind, final_pred


def top_k_candidate(prob, K=10, num_samples=30):
    """
    sample the top-k candidate from the probability distribution
    prob: shape [length of the string, num_classes], probability distribution
    K: the number of candidates
    Return:
        p_y_prod: [K], the joint probability
        p_y: [K, L], all probabilities
        p_y_ind: [K, L], correspoding index of p_y
    """
    p_y, p_y_ind, p_y_prod = [], [], []
    for i in range(num_samples):
        # sample as multinomial distribution
        ind = torch.multinomial(prob, 1)
        sample = torch.gather(prob, 1, ind).squeeze(dim=-1)
        prod = sample.prod(dim=0).item()
        if prod in p_y_prod:
            continue
        else:
            p_y.append(sample)
            p_y_ind.append(ind.squeeze(dim=-1))
            p_y_prod.append(prod)

    p_y = torch.stack(p_y, dim=0)
    p_y_ind = torch.stack(p_y_ind, dim=0)
    p_y_prod = p_y.prod(dim=1)

    k_prod, k_ind = torch.topk(p_y_prod, min(K, len(p_y_prod)), dim=0)

    return p_y_prod[k_ind], p_y[k_ind, ...], p_y_ind[k_ind, ...]


def top_k_candidate_v2(prob, K=10, num_samples=30, K2=5):
    """
    sample the top-k candidate from the probability distribution.
    In this version, we first choose the top-K samples of each prediciton of char, then
    sample from the choosed top-K predictios. 
    prob: shape [length of the string, num_classes], probability distribution
    K: the number of candidates
    K2: the number of choosed top-K candidates
    """
    if K2 <= 0 or K2 >= prob.shape[-1]:
        return top_k_candidate(prob, K, num_samples)
    else:
        # [L, C] -> [L, K2]
        k_prob, k_ind = torch.topk(prob, k=K2, dim=-1)
        # k_p_y_ind: [K, L]
        k_p_y_prod, k_p_y, k_p_y_ind = top_k_candidate(k_prob, K, num_samples)
        # map k_p_y_ind -> k_ind
        p_y_ind = torch.gather(k_ind, 1, k_p_y_ind.transpose(0, 1)).transpose(0, 1)

        return k_p_y_prod, k_p_y, p_y_ind


if __name__ == "__main__":
    # sum_x = 0
    # for i in range(1000):
    #     a = torch.tensor([  [0.1, 0.2, 0.7],
    #                         [0.2, 0.7, 0.1] ], dtype=torch.float)
    #     x = torch.multinomial(a, 1)
    #     # print(x)
    #     sum_x += x

    # print(sum_x / 1000)
    probs = torch.tensor([  [0.1, 0.2, 0.7],
                            [0.2, 0.7, 0.1],
                            [0.7, 0.1, 0.2] ], dtype=torch.float)
    # probs = torch.tensor([  [0.1, 0.2, 0.7] ], dtype=torch.float)
    p, p_y, p_y_ind = top_k_candidate_v2(probs, K=4, num_samples=30, K2=2)
    print("input probability:\n", probs)
    print("joint probability:\n", p)
    print("probabilities of discrete random variable:\n", p_y)
    print("index:\n", p_y_ind)
