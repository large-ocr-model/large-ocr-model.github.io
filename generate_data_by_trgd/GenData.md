## MJST+ DataSet Generation
We leveraged the [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) and [SynthText](https://github.com/ankush-me/SynthText) to generate the MJST+ dataset, with the specific generation process outlined as follows.
### 1. Preparation

1. list_gallery: Randomly selected 700,000 text corpora were obtained from [corpora](https://www.english-corpora.org/corpora.asp).
2. background imgs: Acquired 8,000 natural scene images as backgrounds from [SynthText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c).

### 2. Installation
Install the pypi package
```shell
 cd generate_data_by_trgd/TextRecognitionDataGenerator
 pip install -r requirements.txt
```

### 3. Run for gan data


For convenience, assume that you have placed the corpora data in the `raw_gallery_data_folder`. Use the following script to generate a `specified number` of `list_gallery.txt`.
```shell
list_gallery_path=/path/to/your/list_gallery.txt
count=60000000 # to generate a specified quantity of corpus data.
python gen_gallery.py $raw_gallery_data_folder $list_gallery_path $count
```
Assume that you have placed the background data in `BG_IMG_PATH`, then run the following script to generate the data.
```shell
sh run_gen_data.sh
```
The final format of the generated data is as follows:
```none
.
├── data
│   ├── Label_data.txt
│   ├── Label_data_clean.txt
│   └── imgs
└── labels
    └── details
        └── list_gallery.txt
```

### Acknowledgement

We are grateful for the generation tools provided by the [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) and [SynthText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c).
