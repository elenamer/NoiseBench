# NoiseBench

## About

NoiseBench serves for benchmarking the impact of real label noise on named entity recognition. It is based on a subset of the English CoNLL-03 dataset and consists of 1 ground-truth label set and 6 variants of noisy labels:

- Clean
- Expert noise
- Crowd noise
- Crowd noise (best-case)
- Distant supervision noise
- Weak supervision noise
- LLM noise

We provide the annotation-only files in ```data/annotations```. The annotations follow the IOB2 scheme. We masked the tokens in the included sentences with [TOK] due to the license of the Reuters Corpus that CoNLL-03 is based on. We take the [CleanCoNLL](https://aclanthology.org/2023.emnlp-main.533.pdf) annotations as a ground truth. 

## Instructions

### Create the NoiseBench datasets

This script generates the NoiseBench dataset variants in ``data/noisebench``.

#### Option 1

1. Run the script:

``` 
bash create_noisebench.sh
```

#### Option 2 
(if the ```git clone``` command from Option 1 is not available)

1. Download the full CleanCoNLL dataset in the ```data/cleanconll``` folder according to the instructions in https://github.com/flairNLP/CleanCoNLL.git.

2. Create noisy datasets
``` 
python scripts/generate_data_files.py
```

### Running experiments

1. Requirements
```
conda create -name noisebench python=3.10
conda activate noisebench
pip install -r requirements.txt
```

2. Run
```
python main.py --config exp1_real_noise.json
```
