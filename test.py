#!/usr/bin/env python3

import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import List
import string
import torch
from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |', file=file)
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
          f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |', file=file)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')",
                        default="/path/to/your/clip4str_b_plus.ckpt") 
    parser.add_argument('--data_root', default='/path/to/your/dataset')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--clip_model_path', type=str,
                        default="./clip/ViT-B-16.pt",
                        help='path to the clip model')
    parser.add_argument('--clip_refine', action='store_true', default=False,
                        help='use clip to refine the predicted results')
    parser.add_argument('--sample_K', type=int, default=5,
                        help='K of top-K when performing CLIP post-refinement')
    parser.add_argument('--sample_K2', type=int, default=5,  
                        help='K of top-K choosed predictions when performing CLIP post-refinement')
    parser.add_argument('--sample_total', type=int, default=50,
                        help='the number of samples when sample from the predicted probability distribution')
    parser.add_argument('--sample_prompt', type=str, default=None,
                        help='prompt for CLIP')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='prompt for CLIP')

    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(args)

    if os.path.isdir(args.checkpoint):
        ckpts = [x for x in os.listdir(args.checkpoint) if 'val' in x]
        assert len(ckpts) >= 1
        val_acc = [float(x.split('-')[-2].split('=')[-1]) for x in ckpts]
        best_ckpt = os.path.join(args.checkpoint, ckpts[val_acc.index(max(val_acc))])
        best_epoch = int(best_ckpt.split('/')[-1].split('-')[0].split('=')[-1])
        print('The val accuracy is best {}-{}e'.format(max(val_acc), best_epoch))
        args.checkpoint = best_ckpt

    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    print('load weights from checkpoint {}'.format(args.checkpoint))
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams

    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)

    if args.new:
        test_set = SceneTextDataModule.TEST_NEW
    else:
        test_set = SceneTextDataModule.TEST_BENCHMARK_SUB
    test_set = sorted(set(test_set))

    start_time = time.time()
    results = {}
    max_width = max(map(len, test_set))
    all_total = 0
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((imgs.to(model.device), labels), -1,
                                  clip_model_path=args.clip_model_path,
                                  clip_refine=args.clip_refine,
                                  sample_K=args.sample_K,
                                  sample_K2=args.sample_K2,
                                  sample_total=args.sample_total,
                                  sample_prompt=args.sample_prompt,
                                  alpha=args.alpha
                                  )['output']

            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length
        all_total += total
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)


    if args.new:
        result_groups = {
            'Union14M-Benchmark (Subset)': SceneTextDataModule.TEST_NEW,
        }
    else:
        result_groups = {
            'Six-Common-Benchmark (Subset)': SceneTextDataModule.TEST_BENCHMARK_SUB,
        }
    total_time = time.time() - start_time

    if args.clip_refine:
        log_filename = args.checkpoint + '.log_K{}-{}-{}_prompt{}_alpha{}_new{}.txt'.format(
            args.sample_K2, args.sample_K, args.sample_total, args.sample_prompt, args.alpha, str(args.new))
    else:
        log_filename = args.checkpoint + '.log_new{}.txt'.format(str(args.new))

    with open(log_filename, 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)
            print("Time: Total {}s, Average {}ms. Total samples {}.".format(total_time, total_time * 1000.0 / all_total,
                                                                            all_total), file=out)

if __name__ == '__main__':
    main()
