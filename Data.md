# Source Datasets of REBU-Syn

- We collected labeled data from 16 publicly available real datasets to construct REBU-Syn. The details of these datasets are listed in the following table.


| Data file name | Size | Link | License |
| --- | ---: | --- | --- |
| OpenVINO | 1.5M | https://storage.googleapis.com/openimages/web/index.html | [Apache License 2.0](https://github.com/openvinotoolkit/training_extensions/blob/develop/LICENSE) |
| TextOCR | 0.8M | https://textvqa.org/textocr/dataset/ | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| ICDAR2013 | 843 | https://rrc.cvc.uab.es/?ch=2 | Unknown |
| ICDAR2015 | 4,467 | https://rrc.cvc.uab.es/?ch=4 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)    |
| IIIT5K | 2,000 | https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset | [MIT License](https://github.com/twbs/bootstrap/blob/main/LICENSE) |
| SVT | 257 | http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset | Unknown |
| Total-Text | 12,251 | https://github.com/cs-chan/Total-Text-Dataset | [BSD-3 license](https://github.com/cs-chan/Total-Text-Dataset/blob/master/LICENSE) |
| CTW1500 | 3,170 | https://github.com/Yuliang-Liu/Curve-Text-Detector | Unknown |
| Uber                   | 127,850 | https://s3-us-west-2.amazonaws.com/uber-common-public/ubertext/index.html | Unknown                                                      |
| RCTW17                 |  10,245 | https://rctw.vlrlab.net/dataset                              | Unknown                                                      |
| COCOv2.0 | 72,950 | https://vision.cornell.edu/se3/coco-text-2/ | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| LSVT | 8,164 | https://rrc.cvc.uab.es/?ch=16 | Unknown |
| MLT19 | 55,112 | https://rrc.cvc.uab.es/?ch=15 | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| ReCTS | 26,040 | https://rrc.cvc.uab.es/?ch=12 | Unknown |
| ArT | 31,966 | https://rrc.cvc.uab.es/?ch=14 | Unknown |
| Union14M_L_lmdb_format | 3M | https://github.com/Mountchicken/Union14M/tree/main?tab=readme-ov-file#34-download | [MIT License](https://github.com/Mountchicken/Union14M/blob/main/LICENSE) |

- We collected labeled data from 4 publicly available synthetic datasets to construct REBU-Syn. The details of these datasets are listed in the following table.


| Data file name   | Size | Link                                                         | License                                                      |
| ---------------- | ---: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MJ               |   6M | https://www.robots.ox.ac.uk/~vgg/data/text/                  | Unknown                                                      |
| ST               |   9M | https://www.robots.ox.ac.uk/~vgg/data/scenetext/             | Unknown                                                      |
| Curved SynthText | 1.7M | https://github.com/Jyouhou/ICDAR2019-ArT-Recognition-Alchemy | [Apache License 2.0](https://github.com/PkuDavidGuan/CurvedSynthText/blob/master/LICENSEgithub.com/openimages/dataset/blob/main/LICENSE) |
| SynthAdd         | 1.2M | https://github.com/wangpengnorman/SAR-Strong-Baseline-for-Text-Recognition | Unknown                                                      |

### Datasets

Download the training dataset from the following links:

1. [LMDB archives](https://drive.google.com/drive/folders/1NYuoi7dfJVgo-zUJogh8UQZgIMpLviOE) for MJ, ST, IIIT5k, SVT, SVTP, IC13, IC15, CUTE80, ArT, RCTW17, ReCTS, LSVT, MLT19, COCO-Text, and Uber-Text.
2. [LMDB archives](https://drive.google.com/drive/folders/1D9z_YJVa6f-O0juni-yG5jcwnhvYw-qC) for TextOCR and OpenVINO.
3. [LMDB archives](https://1drv.ms/u/s!AotJrudtBr-K7xAHjmr5qlHSr5Pa?e=LJRlKQ) for Union14M_L_lmdb_format.
4. [CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)
5. [Total-Text](https://drive.google.com/file/d/1bC68CzsSVTusZVvOkk7imSZSbgD1MqK2/view?usp=sharing)
6. [SynthAdd](https://aistudio.baidu.com/datasetdetail/138433)
7. [Curved SynthText](https://github.com/Jyouhou/ICDAR2019-ArT-Recognition-Alchemy?tab=readme-ov-file)

Then, organize the data as follows:

```none
├── REBU-Syn
├── train
│   └── synth_and_real
│       ├── Curved_SynthText
│       │   ├── syntext1
│       │   └── syntext2
│       ├── SynthAdd
│       │   ├── data.mdb
│       │   └── lock.mdb
│       ├── Union14M_L_lmdb_format
│       │   ├── difficult
│       │   ├── hard
│       │   ├── hell
│       │   ├── medium
│       │   └── simple
│       ├── benchmark
│       │   ├── ICDAR2013
│       │   ├── ICDAR2015
│       │   ├── IIIT5K
│       │   └── SVT
│       ├── extra
│       │   ├── CTW1500
│       │   └── total_text
│       └── real_data
│       │   ├── ArT
│       │   ├── COCOv2.0
│       │   ├── LSVT
│       │   ├── MLT19
│       │   ├── OpenVINO
│       │   ├── RCTW17
│       │   ├── ReCTS
│       │   ├── TextOCR
│       │   └── Uber
│       └── mj_st
│           ├── data.mdb
│           └── lock.mdb
└── val
│   ├── CUTE80
│   ├── IC13_1015
│   ├── IC15_1811
│   ├── IIIT5k
│   ├── SVT
│   └── SVTP
├── test
│   ├── CUTE80
│   ├── IC13_1015
│   ├── IC13_857
│   ├── IC15_1811
│   ├── IIIT5k
│   ├── SVT
│   └── SVTP

```
### Data Generation
We generated MJST+(60M) using TextRecognitionDataGenerator and SynthText. For specific generation methods, please refer to [GenData.md](https://github.com/large-ocr-model/large-ocr-model.github.io/blob/main/generate_data_by_trgd/GenData.md)
### Acknowledgement

We sincerely thank all the constructors of the 20 datasets used in REBU-Syn.

- [PARSeq](https://huggingface.co/docs/transformers/main/model_doc/blip-2): the dataset we built upon. Thanks for their wonderful work!
- [Union14M](https://github.com/Mountchicken/Union14M/tree/main): organizes a challenging STR training data. Don't forget to check this great open-source work if you don't know it before!

