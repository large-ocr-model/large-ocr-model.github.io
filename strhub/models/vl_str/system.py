# coding=utf-8
import os
import math
import numpy as np
from itertools import permutations
from typing import Sequence, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.types import STEP_OUTPUT

from strhub.clip import clip
from strhub.models.base import CrossEntropySystem
from .modules import DecoderLayer, Decoder, modify_attn_mask


# an alternative choice when the input argument is not valid 
CLIP_PATH = '/PUT/YOUR/PATH/HERE/pretrained/clip'


class VL4STR(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
                 decode_ar: bool, refine_iters: int, dropout: float, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters
        self.coef_lr = kwargs["coef_lr"] if "coef_lr" in kwargs.keys() else 1.0
        self.coef_wd = kwargs["coef_wd"] if "coef_wd" in kwargs.keys() else 1.0
        self.image_freeze_nlayer = kwargs["image_freeze_nlayer"] if "image_freeze_nlayer" in kwargs.keys() else -1
        self.text_freeze_nlayer = kwargs["text_freeze_nlayer"] if "text_freeze_nlayer" in kwargs.keys() else -1
        self.freeze_language_backbone = self.text_freeze_nlayer >= 12
        self.freeze_image_backbone = self.image_freeze_nlayer >= 12
        self.use_language_model = kwargs["use_language_model"] if "use_language_model" in kwargs.keys() else False
        self.context_length = kwargs["context_length"] if "context_length" in kwargs.keys() else 20
        self.cross_loss_w = kwargs["cross_loss_w"] if "cross_loss_w" in kwargs.keys() else 1.0
        self.use_share_dim = kwargs["use_share_dim"] if "use_share_dim" in kwargs.keys() else True
        self.cross_gt_context = kwargs["cross_gt_context"] if "cross_gt_context" in kwargs.keys() else True
        self.cross_cloze_mask = kwargs["cross_cloze_mask"] if "cross_cloze_mask" in kwargs.keys() else False
        self.cross_correct_once = kwargs["cross_correct_once"] if "cross_correct_once" in kwargs.keys() else False
        self.image_detach = kwargs["image_detach"] if "image_detach" in kwargs.keys() else True
        self.cross_token_embeding = kwargs["cross_token_embeding"] if "cross_token_embeding" in kwargs.keys() else False
        self.cross_fast_decode = False

        rank_zero_info("\n config of VL4STR: \n"
                "\t image_freeze_nlayer: {}, text_freeze_nlayer: {}, freeze_language_backbone: {}, freeze_image_backbone: {} \n"
                "\t use_language_model: {}, context_length: {}, cross_token_embeding: {}, cross_loss_weight: {} \n"
                "\t use_share_dim: {}, image_detach: {} \n"
                "\t cross_gt_context: {}, cross_cloze_mask: {}, cross_fast_decode: {} \n".format(
                self.image_freeze_nlayer, self.text_freeze_nlayer, self.freeze_language_backbone, self.freeze_image_backbone,
                self.use_language_model, self.context_length,  self.cross_token_embeding, self.cross_loss_w,
                self.use_share_dim, self.image_detach, self.cross_gt_context, self.cross_cloze_mask, self.cross_fast_decode)
                )

        assert "clip_pretrained" in kwargs.keys()
        if not os.path.exists(kwargs["clip_pretrained"]):
            kwargs["clip_pretrained"] = os.path.join(CLIP_PATH, os.path.basename(kwargs["clip_pretrained"]))
            assert os.path.exists(kwargs["clip_pretrained"])
        # load CLIP model
        clip_model, _ = clip.load(name=kwargs["clip_pretrained"], device='cpu')
        self.clip_model = clip_model.float()

        # modify the attention mask according to context length
        self.clip_model.transformer.apply(lambda m: modify_attn_mask(m, context_length=self.context_length))
        self.freeze_cip_layers(self.image_freeze_nlayer, self.text_freeze_nlayer)

        # visual deoder
        vis_embed_dim = self.clip_model.text_projection.shape[-1] if self.use_share_dim else self.clip_model.visual.proj.shape[0]
        rank_zero_info("The dimension of the visual decoder is {}.".format(vis_embed_dim))
        decoder_layer = DecoderLayer(vis_embed_dim, dec_num_heads, vis_embed_dim * dec_mlp_ratio, dropout)
        # We don't predict <bos> nor <pad>, so num_classes=len(self.tokenizer) - 2, label length + 1 for <eos>
        self.visual_decoder = Decoder(decoder_layer, dec_depth, norm=nn.LayerNorm(vis_embed_dim),
                                        embed_dim=vis_embed_dim, 
                                        dropout=dropout,
                                        num_classes=len(self.tokenizer) - 2,
                                        charset_size=len(self.tokenizer),
                                        max_label_length=max_label_length + 1)
        # cross-modal decoder
        if self.use_language_model:
            cross_embed_dim = self.clip_model.text_projection.shape[-1]
            decoder_layer = DecoderLayer(cross_embed_dim, dec_num_heads, cross_embed_dim * dec_mlp_ratio, dropout)
            self.cross_decoder = Decoder(decoder_layer, dec_depth, norm=nn.LayerNorm(cross_embed_dim),
                                        embed_dim=cross_embed_dim,
                                        dropout=dropout,
                                        num_classes=len(self.tokenizer) - 2,
                                        charset_size=len(self.tokenizer),
                                        max_label_length=max_label_length + 1)
        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

    def encode(self, img: torch.Tensor):
        """extract CLIP image features"""
        if self.freeze_image_backbone:
            with torch.no_grad():
                memory = self.clip_model.encode_image(img, projection=self.use_share_dim)
        else:
            memory = self.clip_model.encode_image(img, projection=self.use_share_dim)

        return memory

    def visual_decode(self, tgt: torch.Tensor, memory: torch.Tensor,
                        tgt_query: Optional[Tensor] = None, tgt_query_mask: Optional[Tensor] = None,
                        content_mask: Optional[Tensor] = None, tgt_padding_mask: Optional[Tensor] = None, ):
        return self.visual_decoder(tgt, memory, tgt_query, tgt_query_mask, content_mask, tgt_padding_mask)

    def encoder_cross_modal_feature(self, prev_logits, image_feat):
        prev_logits = prev_logits.detach().clone()
        image_features = image_feat.detach().clone() if self.image_detach else image_feat
        if not self.use_share_dim:
            image_features = torch.matmul(image_features, self.clip_model.visual.proj)

        # get previous predictions
        probs = prev_logits.softmax(-1)
        # adapt for the test charset, CLIP is not sensitive to uppercase or symbols
        captions, _ = self.tokenizer.decode_fast(probs, charset_adapter=self.charset_adapter)
        text = clip.tokenize(captions, context_length=self.context_length, truncate=True).to(image_feat.device)

        # return all text features
        if self.freeze_language_backbone:
            with torch.no_grad():
                text_features = self.clip_model.token_embedding(text) if self.cross_token_embeding else \
                                    self.clip_model.encode_text(text)
        else:
            text_features = self.clip_model.token_embedding(text) if self.cross_token_embeding else \
                                    self.clip_model.encode_text(text)        

        return torch.cat([image_features, text_features], dim=1)

    def cross_decode(self, prev_logits, tgt: torch.Tensor, memory: torch.Tensor,
                        tgt_query: Optional[Tensor] = None, tgt_query_mask: Optional[Tensor] = None, 
                        content_mask: Optional[Tensor] = None, tgt_padding_mask: Optional[Tensor] = None,
                        cross_memory = None):
        if cross_memory is None:
            cross_memory = self.encoder_cross_modal_feature(prev_logits, memory)

        return self.cross_decoder(tgt, cross_memory, tgt_query, tgt_query_mask, content_mask, tgt_padding_mask)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)

        # Query positions up to `num_steps`
        vis_pos_queries = self.visual_decoder.pos_queries[:, :num_steps].expand(bs, -1, -1)
        crs_pos_queries = self.cross_decoder.pos_queries[:, :num_steps].expand(bs, -1, -1) if self.use_language_model else None

        # a left-to-right auto-regressive mask, special case for the forward permutation
        content_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)
        bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.bos_id

            logits = []
            all_visual_vec = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding: Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                p_i, visual_vec = self.visual_decode(tgt_in[:, :j], memory, tgt_query=vis_pos_queries[:, i:j],
                                            tgt_query_mask=query_mask[i:j, :j], content_mask=content_mask[:j, :j],)
                # the next token probability is in the output's ith token position
                logits.append(p_i)
                all_visual_vec.append(visual_vec.clone())
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break
            logits = torch.cat(logits, dim=1)
            visual_vec = torch.cat(all_visual_vec, dim=1)

        else:
            # No prior context, so input is just <bos>. We query all positions.
            # tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            logits, visual_vec = self.visual_decode(bos, memory, tgt_query=vis_pos_queries)

        if self.use_language_model:
            crs_num_steps = logits.shape[1]
            if self.cross_fast_decode:
                # just use visual output as input context
                cross_logits, cross_vec = self.cross_decode(logits, tgt_in[:, :crs_num_steps], memory, tgt_query=crs_pos_queries[:, :crs_num_steps],
                                            tgt_query_mask=query_mask[:crs_num_steps, :crs_num_steps],
                                            content_mask=content_mask[:crs_num_steps, :crs_num_steps],)
            else:
                # prediction of cross-modal branch as input context
                cross_memory = self.encoder_cross_modal_feature(logits, memory)
                cross_logits = []
                all_cross_vec = []
                for i in range(crs_num_steps):
                    j = i + 1  # next token index
                    p_i, cross_vec = self.cross_decode(logits, tgt_in[:, :j], memory, tgt_query=crs_pos_queries[:, i:j],
                                                tgt_query_mask=query_mask[i:j, :j], content_mask=content_mask[:j, :j], cross_memory=cross_memory)                
                    cross_logits.append(p_i)
                    all_cross_vec.append(cross_vec.clone())
                    if j < crs_num_steps:
                        tgt_in[:, j] = p_i.squeeze().argmax(-1)
                cross_logits = torch.cat(cross_logits, dim=1)
                cross_vec = torch.cat(all_cross_vec, dim=1)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = ((tgt_in == self.eos_id).cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
                logits, visual_vec = self.visual_decode(tgt_in, memory,
                                                    tgt_query=vis_pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]],
                                                    content_mask=content_mask, tgt_padding_mask=tgt_padding_mask,)
                if self.use_language_model:
                    tgt_in = torch.cat([bos, cross_logits[:, :-1].argmax(-1)], dim=1)
                    tgt_padding_mask = ((tgt_in == self.eos_id).cumsum(-1) > 0)
                    cross_logits, cross_vec = self.cross_decode(logits, tgt_in, memory,
                                                    tgt_query=crs_pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]],
                                                    content_mask=content_mask, tgt_padding_mask=tgt_padding_mask,)

        # TODO: how to fuse the final predictions
        logits = cross_logits
        return logits

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device)

        # Encode the source sequence (i.e. the image codes)
        memory = self.encode(images)

        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]        # remove [EOS] token
        tgt_out = tgt[:, 1:]        # remove [BOS] token
        # cross_target = tgt[:, 1:]
        # max_len = min(cross_target.shape[1] - 1, self.max_label_length)
        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
        bs = images.shape[0]
        bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
        # try to use a cloze_mask for cross decoding
        L = tgt_in.shape[1]
        cloze_content_mask = cloze_query_mask = torch.triu(torch.full((L, L), float('-inf'), device=self._device), 1)
        cloze_query_mask[torch.triu(torch.ones(L, L, dtype=torch.bool, device=self._device), 2)] = 0

        loss = 0
        loss_numel = 0
        all_inter_vars = []
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            content_mask, query_mask = self.generate_attn_masks(perm)
            visual_logits, visual_vec = self.visual_decode(tgt_in, memory, tgt_query_mask=query_mask,
                                                content_mask=content_mask, tgt_padding_mask=tgt_padding_mask, )
            loss += n * F.cross_entropy(visual_logits.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)
            all_inter_vars.append((visual_logits.detach().clone(), query_mask, content_mask))

            # cross forward
            if self.use_language_model:
                # 1. use the prediction of visual branch as context, keep the shape of cross_tgt_in the same as `tgt_in`
                # 2. use the GT as the context
                cross_tgt_in = tgt_in if self.cross_gt_context else torch.cat([bos, visual_logits[:, :-1].argmax(-1)], dim=1)
                cross_query_mask = cloze_query_mask if self.cross_cloze_mask else query_mask
                cross_content_mask = cloze_content_mask if self.cross_cloze_mask else content_mask
                cross_logits, cross_vec = self.cross_decode(visual_logits, cross_tgt_in, memory, tgt_query_mask=cross_query_mask,
                                                content_mask=cross_content_mask, tgt_padding_mask=tgt_padding_mask, )
                loss += self.cross_loss_w * n * F.cross_entropy(cross_logits.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)

            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()

        loss /= loss_numel

        self.log('loss', loss)
        return loss

    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
           This works because the same attention mask can be used for the shorter sequences
           because of the padding mask.
           An example fo perms with string length 5
           >>> tensor([[0, 1, 2, 3, 4, 5, 6],   # canonical order
                    [0, 6, 5, 4, 3, 2, 1],      # reverse order
                    [0, 3, 1, 5, 2, 4, 6],
                    [0, 1, 4, 2, 3, 5, 6],
                    [0, 2, 3, 5, 1, 4, 6],
                    [0, 2, 1, 3, 5, 4, 6]])
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequences and shorter, we generate all permutations and sample from the pool to avoid collisions
        # Note that this code path might NEVER get executed since the labels in a mini-batch typically exceed 4 chars.
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=self._device)[selector]
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)
        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            # Stack in such a way that the pairs are next to each other.
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # NOTE:
        # The only meaningful way of permuting the EOS position is by moving it one character position at a time.
        # However, since the number of permutations = T! and number of EOS positions = T + 1, the number of possible EOS
        # positions will always be much less than the number of permutations (unless a low perm_num is set).
        # Thus, it would be simpler to just train EOS using the full and null contexts rather than trying to evenly
        # distribute it across the chosen number of permutations.
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for the reverse direction. This does two things:
        # 1. Reverse context for the characters
        # 2. Null context for [EOS] (required for learning to predict [EOS] in NAR mode)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def configure_optimizers(self):
        agb = self.trainer.accumulate_grad_batches
        # Linear scaling so that the effective learning rate is constant regardless of the number of GPUs used with DDP.
        # lr_scale = agb * math.sqrt(self.trainer.num_devices) * self.batch_size / 256.
        lr_scale = agb * (self.trainer.num_devices * self.batch_size) / 512
        lr = lr_scale * self.lr

        # https://github.com/mlfoundations/open_clip/blob/b4cf9269b0b11c0eea47cb16039369a46bd67449/src/training/main.py#L171
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n \
                            or "pos_queries" in n or "text_embed" in n
        include = lambda n, p: not exclude(n, p)

        # encoder parameters
        encoder_params = list(self.clip_model.named_parameters())
        enc_gain_or_bias_params = [p for n, p in encoder_params if exclude(n, p) and p.requires_grad]
        enc_rest_params = [p for n, p in encoder_params if include(n, p) and p.requires_grad]

        # decoder parameters
        decoder_params = [(n, p) for n, p in list(self.named_parameters()) if "clip_model" not in n]
        dec_gain_or_bias_params = [p for n, p in decoder_params if exclude(n, p) and p.requires_grad]
        dec_rest_params = [p for n, p in decoder_params if include(n, p) and p.requires_grad]

        rank_zero_info("[VL4STR] The length of encoder params with and without weight decay is {} and {}, respectively.".format(
            len(enc_rest_params), len(enc_gain_or_bias_params)
        ))
        rank_zero_info("[VL4STR] The length of decoder params with and without weight decay is {} and {}, respectively.".format(
            len(dec_rest_params), len(dec_gain_or_bias_params)
        ))

        optimizer = torch.optim.AdamW(
            [
                {"params": enc_gain_or_bias_params, "weight_decay": 0., 'lr': lr},
                {"params": enc_rest_params, "weight_decay": self.weight_decay, 'lr': lr},
                {"params": dec_gain_or_bias_params, "weight_decay": 0., 'lr': lr * self.coef_lr},
                {"params": dec_rest_params, "weight_decay": self.weight_decay * self.coef_wd, 'lr': lr * self.coef_lr},
            ],
            lr=lr, betas=(0.9, 0.98), eps=1.0e-6,
            )
        sched = OneCycleLR(optimizer, [lr, lr, lr * self.coef_lr, lr * self.coef_lr],
                            self.trainer.estimated_stepping_batches, pct_start=self.warmup_pct,
                            cycle_momentum=False)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    def _freeze_backbones(self):
        """set frozen backbones to eval mode"""
        for name, mod in self.clip_model.named_modules():
            if name.startswith("visual.transformer.resblocks."):
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num < self.image_freeze_nlayer:
                    mod.eval()

                # if self.image_freeze_layer_divisor > 0 and (layer_num + 1) % self.image_freeze_layer_divisor == 0:
                #     mod.eval()
            elif name.startswith("transformer.resblocks."):
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num < self.text_freeze_nlayer:
                    mod.eval()

        if self.freeze_language_backbone:
            self.clip_model.transformer.eval()
            self.clip_model.ln_final.eval()

    def freeze_cip_layers(self, image_freeze_nlayer, text_freeze_nlayer, image_freeze_layer_divisor=-1, image_only_fc=False):
        """
        freeze the parameters of layers with No.layer < image_freeze_nlayer or text_freeze_nlayer,
        """
        assert image_freeze_nlayer <= 12 and text_freeze_nlayer <=12 and image_freeze_layer_divisor <= 12
        if hasattr(self, "clip_model"):
            if image_freeze_nlayer > -1:
                for name, param in self.clip_model.visual.named_parameters():
                    # top layers always need to train
                    if name.startswith("ln_post.") or name.startswith("proj") or name.startswith("conv1") or name.startswith("ln_pre"):
                        continue
                    elif name.startswith("transformer.resblocks."):
                        layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                        if layer_num >= image_freeze_nlayer:
                            continue
                    param.requires_grad = False

            #### freeze the layers which the index can be divided by image_freeze_layer_divisor
            # if image_freeze_layer_divisor > 0:
            #     for name, param in self.clip_model.visual.named_parameters():
            #         if name.startswith("transformer.resblocks."):
            #             layer_num = int(name.split(".resblocks.")[1].split(".")[0])
            #             if (layer_num + 1) % image_freeze_layer_divisor == 0:
            #                 param.requires_grad = False

            #### only train the top fc layer
            # if image_only_fc:
            #     for name, param in self.clip_model.visual.named_parameters():
            #         if "out_proj" in name or "conv1" in name or name.startswith("ln_post.") or name.startswith("proj"):
            #             continue
            #         param.requires_grad = False

            if text_freeze_nlayer > -1:
                for name, param in self.clip_model.named_parameters():
                    # top layers always need to train
                    if name.startswith("ln_final.") or name.startswith("text_projection") or name.startswith("visual"):
                        continue
                    elif name.startswith("transformer.resblocks."):
                        layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                        if layer_num >= text_freeze_nlayer:
                            continue
                    param.requires_grad = False

            # freeze the whole backbones and related parameters
            if text_freeze_nlayer >= 12:
                for n, p in self.clip_model.named_parameters():
                    # exclude visual parameters
                    if "visual" not in n:                
                        if "transformer" in n or "token_embedding" in n or "ln_final" in n or "text_projection" in n:
                            p.requires_grad = False
            
            if image_freeze_nlayer >= 12:
                for n, p in self.clip_model.visual.named_parameters():
                    p.requires_grad = False               

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if mode:
            self._freeze_backbones()

        return self
