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

import pytorch_lightning as pl
from pathlib import PurePath
from torch.utils.data import DataLoader
from torchvision import transforms as T
from typing import Optional, Callable, Sequence, Tuple
from pytorch_lightning.utilities import rank_zero_info

from .dataset import LmdbDataset


class SceneTextDataModule(pl.LightningDataModule):
    TEST_BENCHMARK_SUB = ('IIIT5k', 'SVT', 'IC13_857', 'IC15_1811', 'SVTP', 'CUTE80')
    TEST_NEW = ('artistic', 'contextless', 'curve', 'general', 'multi_oriented', 'multi_words', 'salient') #

    TEST_ALL = tuple(set(TEST_BENCHMARK_SUB))
    def __init__(self, root_dir: str, train_dir: str, img_size: Sequence[int], max_label_length: int,
                 charset_train: str, charset_test: str, batch_size: int, num_workers: int, augment: bool,
                 remove_whitespace: bool = False, normalize_unicode: bool = False,
                 min_image_dim: int = 0, rotation: int = 0, collate_fn: Optional[Callable] = None,
                 output_url: str = None, openai_meanstd: bool = True,):
        super().__init__()
        self.root_dir = root_dir
        self.train_dir = train_dir
        self.img_size = tuple(img_size)
        self.max_label_length = max_label_length
        self.charset_train = charset_train
        self.charset_test = charset_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.remove_whitespace = remove_whitespace
        self.normalize_unicode = normalize_unicode
        self.min_image_dim = min_image_dim
        self.rotation = rotation
        self.collate_fn = collate_fn
        self._train_dataset = None
        self._val_dataset = None

        self.mean = (0.48145466, 0.4578275, 0.40821073) if openai_meanstd else 0.5
        self.std = (0.26862954, 0.26130258, 0.27577711) if openai_meanstd else 0.5
        rank_zero_info("[dataset] mean {}, std {}".format(self.mean, self.std))

    @staticmethod
    def get_transform(img_size: Tuple[int], augment: bool = False, rotation: int = 0, mean=0.5, std=0.5):
        transforms = []
        if rotation:
            transforms.append(lambda img: img.rotate(rotation, expand=True))
        transforms.extend([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        return T.Compose(transforms)

    def test_dataloaders(self, subset):
        transform = self.get_transform(self.img_size, rotation=self.rotation, mean=self.mean, std=self.std)
        root = PurePath(self.root_dir, 'test')
        datasets = {s: LmdbDataset(str(root / s), self.charset_test, self.max_label_length,
                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                   transform=transform) for s in subset}
        return {k: DataLoader(v, batch_size=self.batch_size, num_workers=self.num_workers,
                              pin_memory=True, collate_fn=self.collate_fn)
                for k, v in datasets.items()}
