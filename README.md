# NoiseBench: Benchmarking the Impact of Real Label Noise on Named Entity Recognition

We created a benchmark for learning with noisy labels. More details can be found in our EMNLP 2024 [paper] (https://arxiv.org/abs/2405.07609). 

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

### Run experiments

1. Requirements
```
conda create -n noisebench python=3.10
conda activate noisebench
pip install -r requirements.txt
```

2. Run main experiment script
```
python main.py --config configs/exp1_real_noise.json
```

### Run simulated noise experiments

1. Run simulated noise generation
```
python scripts/calculate_data_overviews.py
python scripts/create_simulated_noisy_sets.py
```

2. Run main experiment script
```
python main.py --config configs/exp1_simulated_noise.json
```


### Additional experiment: Create the German NoiseBench datasets

This script generates the german version in ``data/noisebench_german``.

1. Get the full dataset following the instructions here: https://www.clips.uantwerpen.be/conll2003/ner/. With this, the full dataset (files ```deu.train```, ```deu.testa```, ```deu.testb```) should be downloaded in the ```data/conll_german``` directory.

2. Run:

``` 
python scripts/generate_german_data_files.py
```
