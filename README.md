# NoiseBench

## About

The following annotation variants are included, as described in the manuscript:

- clean
- noise_expert
- noise_crowd
- noise_crowdbest
- noise_distant
- noise_weak
- noise_llm

NoiseBench is based on a subset of the English CoNLL-03 dataset. The annotations follow the IOB2 scheme. 

In this repository, we only provide the annotation files. we masked the tokens in the included sentences with [TOK] due to the original license of the Reuters Corpus that CoNLL-03 is based on. Additionally, we take the [CleanCoNLL](https://aclanthology.org/2023.emnlp-main.533.pdf) annotations as a reference point.

## Instructions

1. Download cleanconll in data/cleanconll according to the instructions in https://github.com/flairNLP/CleanCoNLL.git.

2. Generate noisy datasets
``` 
python scripts/generate_data_files.py
```