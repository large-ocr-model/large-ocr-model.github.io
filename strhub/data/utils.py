# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
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

import math
import re
from abc import ABC, abstractmethod
from itertools import groupby
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class CharsetAdapter:
    """Transforms labels according to the target charset."""

    def __init__(self, target_charset) -> None:
        super().__init__()
        self.lowercase_only = target_charset == target_charset.lower()
        self.uppercase_only = target_charset == target_charset.upper()
        self.unsupported = f'[^{re.escape(target_charset)}]'

    def __call__(self, label):
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        # Remove unsupported characters
        label = re.sub(self.unsupported, '', label)
        return label


class BaseTokenizer(ABC):

    def __init__(self, charset: str, specials_first: tuple = (), specials_last: tuple = ()) -> None:
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> List[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    @abstractmethod
    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        """Encode a batch of labels to a representation suitable for the model.

        Args:
            labels: List of labels. Each can be of arbitrary length.
            device: Create tensor on this device.

        Returns:
            Batched tensor representation padded to the max label length. Shape: N, L
        """
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        """Internal method which performs the necessary filtering prior to decoding."""
        raise NotImplementedError

    def decode(self, token_dists: Tensor, raw: bool = False) -> Tuple[List[str], List[Tensor]]:
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)

        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of Tensors
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            # probs, ids = beam_search_decoder(dist, 5)
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs

    def decode_for_refine(self, token_dists: Tensor, raw: bool = False) -> Tuple[List[str], List[Tensor]]:
        raise NotImplementedError


class Tokenizer(BaseTokenizer):
    BOS = '[B]'
    EOS = '[E]'
    PAD = '[P]'

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        batch = [torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
                 for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
        return probs, ids

    def decode_fast(self, token_dists: Tensor, raw: bool = False, charset_adapter = None) -> Tuple[List[str], List[Tensor]]:
        """fast version of decode"""
        bs_tokens = []
        bs_probs = []
        batch_probs, batch_ids = token_dists.max(-1)
        batch_ids = batch_ids.tolist()

        for i, ids in enumerate(batch_ids):
            tokens = self._ids2tok(ids, not raw)
            # Truncate after EOS
            tokens = tokens.partition(self.EOS)[0]
            # but include prob. for EOS (if it exists)
            bs_probs.append(batch_probs[i, :len(tokens) + 1])

            if charset_adapter is not None:
                tokens = charset_adapter(tokens)
            bs_tokens.append(tokens)

        return bs_tokens, bs_probs

    def decode_for_refine(self, token_dists: Tensor, raw: bool = False) -> Tuple[List[str], List[Tensor]]:
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)

        Returns:
            # list of string labels (arbitrary length) and
            # their corresponding sequence probabilities as a list of Tensors
            lisft of sequence probabilities
        """
        batch_dist = []
        for dist in token_dists:
            # greedy selection
            max_probs, ids = dist.max(-1)
            if not raw:
                ids = ids.tolist()
                try:
                    eos_idx = ids.index(self.eos_id)
                except ValueError:
                    eos_idx = len(ids)
                # trunucate after EOS
                ids = ids[:eos_idx]
                # do not include EOS
                trunc_dist = dist[:eos_idx, ...]
                # set probability of [EOS] to zero
                trunc_dist[:, self.eos_id] = torch.zeros_like(trunc_dist[:, self.eos_id])
                batch_dist.append(trunc_dist)
            else:
                batch_dist.append(dist)

        return batch_dist


class CTCTokenizer(BaseTokenizer):
    BLANK = '[B]'

    def __init__(self, charset: str) -> None:
        # BLANK uses index == 0 by default
        super().__init__(charset, specials_first=(self.BLANK,))
        self.blank_id = self._stoi[self.BLANK]

    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        # We use a padded representation since we don't want to use CUDNN's CTC implementation
        batch = [torch.as_tensor(self._tok2ids(y), dtype=torch.long, device=device) for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.blank_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        # Best path decoding:
        ids = list(zip(*groupby(ids.tolist())))[0]  # Remove duplicate tokens
        ids = [x for x in ids if x != self.blank_id]  # Remove BLANKs
        # `probs` is just pass-through since all positions are considered part of the path
        return probs, ids


def beam_search_decoder(data, k):
    """a simple beam search algorithm"""
    device = data.device
    data_numpy = data.cpu().numpy()

    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data_numpy:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - math.log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]

    ids = torch.tensor(sequences[0][0], dtype=torch.long, device=data.device)
    probs = data[torch.arange(len(ids), dtype=torch.long, device=device), ids]
    return probs, ids


if __name__ == "__main__":
    _char_94_full = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    t = Tokenizer(_char_94_full)
    for x in _char_94_full:
        print(t._tok2ids(x))
    print(t._itos)
    print(t._stoi)
    print(t.bos_id)
    print(t.eos_id)
    print(t.pad_id)

# {'[E]': 0, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10, 
# 'a': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15, 'f': 16, 'g': 17, 'h': 18, 'i': 19, 'j': 20, 
# 'k': 21, 'l': 22, 'm': 23, 'n': 24, 'o': 25, 'p': 26, 'q': 27, 'r': 28, 's': 29, 't': 30, 
# 'u': 31, 'v': 32, 'w': 33, 'x': 34, 'y': 35, 'z': 36, 
# 'A': 37, 'B': 38, 'C': 39, 'D': 40, 'E': 41, 'F': 42, 'G': 43, 'H': 44, 'I': 45, 'J': 46, 
# 'K': 47, 'L': 48, 'M': 49, 'N': 50, 'O': 51, 'P': 52, 'Q': 53, 'R': 54, 'S': 55, 'T': 56, 
# 'U': 57, 'V': 58, 'W': 59, 'X': 60, 'Y': 61, 'Z': 62, 
# '!': 63, '"': 64, '#': 65, '$': 66, '%': 67, '&': 68, "'": 69, '(': 70, ')': 71, '*': 72, 
# '+': 73, ',': 74, '-': 75, '.': 76, '/': 77, ':': 78, ';': 79, '<': 80, '=': 81, '>': 82, 
# '?': 83, '@': 84, '[': 85, '\\': 86, ']': 87, '^': 88, '_': 89, '`': 90, '{': 91, '|': 92, 
# '}': 93, '~': 94, '[B]': 95, '[P]': 96}
# bos_id 95
# eos_id 0
# pad_id 96