## Getting Started

### Installation

Requires `Python >= 3.8` and `PyTorch >= 1.10`.

```
conda create --name largeocrmodel python==3.8
pip install -r requirements.txt
```

### Data and Model

- For convenient, you can download the six common benchmark dataset from [CDistNet](https://drive.google.com/file/d/1dTI0ipu14Q1uuK4s4z32DqbqF3dJPdkk/view?usp=sharing), and download Union 14M benchmark dataset from [Union14M dataset](https://1drv.ms/u/s!AotJrudtBr-K7xAHjmr5qlHSr5Pa?e=LJRlKQ).

  select organize the `data` directory as follows after downloading all of them:

- <details>
      <summary> Test Data Structure Tree </summary>

      ```
      .
      ├── test
      │   ├── CUTE80
      │   ├── IC13_1015
      │   ├── IC13_857
      │   ├── IC15_1811
      │   ├── IIIT5k
      │   ├── SVT
      │   ├── SVTP
      │   ├── artistic
      │   ├── contextless
      │   ├── curve
      │   ├── general
      │   ├── multi_oriented
      │   ├── multi_words
      │   └── salient

      ```
  </details>



- weights of CLIP-ViT-B/16 pre-trained models can be found in [CLIP-ViT-B/16](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)


## Results

CLIP4STR-B's word accuracy on the Common benchmark

| Method      | Train data | IIIT5K | SVT   | IC13  | IC15  | SVTP  | CUTE  |
| ----------- | ---------- | ------ | ----- | ----- | ----- | ----- | ----- |
| CLIP4STR-B  | MJ+ST      | 97.70  | 95.36 | 96.06 | 87.47 | 91.47 | 94.44 |
| CLIP4STR-B  | Real       | 99.20  | 98.30 | 98.23 | 91.44 | 96.90 | 99.65 |
| CLIP4STR-B* | RESU-syn   | 98.97  | 98.76 | 99.30 | 92.27 | 97.83 | 99.65 |

CLIP4STR-B's word accuracy on the  Union14M benchmark.

| Method      | Train data | Artistic | Contextless | Curve | General | Multi-Oriented | Multi-Words | Salient |
| ----------- | ---------- | -------- | ----------- | ----- | ------- | -------------- | ----------- | ------- |
| CLIP4STR-B  | Real       | 86.5     | 92.2        | 96.3  | 89.9    | 96.1           | 88.9        | 91.2    |
| CLIP4STR-B* | REBU-Syn   | 88.6     | 90.1        | 96.4  | 89.1    | 96.3           | 92.2        | 91.9    |



## Inference

1. Download the CLIP4STR-B* from [BaiduYun](https://pan.baidu.com/s/1GA2E1piNV8Ts1L19swfmBw)(wjk8)
2. Run the following command to inference on test data:

​    Inference CLIP4STR-B* on six common benchmark

```
python test.py --checkpoint /path/to/your/clip4str_b_plus.ckpt --data_root /path/to/your/eval_dataset --clip_model_path /path/to/your/ViT-B-16.pt
```

  Inference CLIP4STR-B* on Union14M benchmark
```
python test.py --checkpoint /path/to/your/clip4str_b_plus.ckpt --data_root /path/to/your/eval_dataset --new --clip_model_path /path/to/your/ViT-B-16.pt
```

### Acknowledgement

We are very grateful to [CLIP4STR](https://github.com/VamosC/CLIP4STR) for providing the inference framework.


