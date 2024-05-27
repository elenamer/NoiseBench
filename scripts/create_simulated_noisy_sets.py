import flair
import torch
import logging
from flair.datasets import ColumnCorpus, CSVClassificationDataset, CSVClassificationCorpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
import os
import argparse
import numpy as np
from collections import defaultdict
import pandas as pd
from typing import List, Dict, Optional
from utils import *

def add_label_noise(
    sentences_list,
    labels: List[str],
    noise_transition_matrix: Optional[Dict[str, List[float]]] = None,
    seed = 42
):
    """Simulates class-dependent label noise.
    Args:
        labels: an array with unique labels of said type (retrievable from label dictionary).
        noise_transition_matrix: provides pre-defined token-level probabilities for label flipping based on the initial
            label value.
        seed: random seed
    """

    corrupted_count = 0
    total_label_count = 0

    if noise_transition_matrix:
        ntm_labels = noise_transition_matrix.keys()

        if set(ntm_labels) != set(labels):
            raise AssertionError(
                "Label values in the noise transition matrix have to coincide with label values in the dataset"
            )

        print("Generating noisy labels. Progress:")
        new_sentences = []
        for sentence in sentences_list:
            new_sentence = []
            if 'DOCSTART' in sentence[0][0]:
                new_sentences.append(sentence)
                continue
            prev_label = 'O'
            for token in sentence:
                new_token = []
                new_token.append(token[0])
                total_label_count += 1
                orig_label = token[1]
                # sample randomly from a label distribution according to the probabilities defined by the noise transition matrix
                np.random.seed(seed)

                #make sure BIO rules are enforced
                if prev_label == 'O':
                    available_labels = []
                    available_p = []
                    for i,l in enumerate(ntm_labels):
                        if not l.startswith('I'):
                            available_labels.append(l)
                            available_p.append(noise_transition_matrix[orig_label][i])
                else:
                    available_labels = []
                    available_p = []
                    for i,l in enumerate(ntm_labels):
                        if l == 'O':
                            available_labels.append(l)
                            available_p.append(noise_transition_matrix[orig_label][i])
                        elif l.startswith('B'):
                            available_labels.append(l)
                            available_p.append(noise_transition_matrix[orig_label][i])
                        else:
                            if l.endswith(prev_label[-3:]):
                                available_labels.append(l)
                                available_p.append(noise_transition_matrix[orig_label][i])
                available_p = np.array(available_p) / sum(np.array(available_p))
                new_label = np.random.default_rng().choice(
                    a=available_labels,
                    p=available_p,
                )
                # replace the old label with the new one
                new_token.append(new_label)
                prev_label=new_label
                # keep track of how many labels in total are flipped
                if new_label != orig_label:
                    corrupted_count += 1
                new_sentence.append(new_token)
            new_sentences.append(new_sentence)

    print(
        f"Total labels corrupted: {corrupted_count}. Resulting noise share: {round((corrupted_count / total_label_count) * 100, 2)}%."
    )
    return new_sentences

# set paths

noisebench_data_path =  'data/noisebench/'
simulated_data_path = f'{noisebench_data_path}simulated/'

train_clean_filename = f'{noisebench_data_path}clean.train'
dev_clean_filename = f'{noisebench_data_path}clean.dev'
all_clean_filename = f'{noisebench_data_path}clean.traindev'

noise_transition_matrix_path = 'results/data_overviews/noisebench/'

corpora = ['noise_expert','noise_crowdbest','noise_crowd','noise_distant','noise_distant','noise_llm']

for corpus_name in corpora:

    # read token-level transition matrix, generated with calculate_data_overviews script
    ntm_df = pd.read_csv(noise_transition_matrix_path+'/NTM_'+corpus_name+'.csv', index_col = 0, header=0)
    labels=ntm_df.index.values
    ntm={}
    for row in ntm_df.iterrows():
        ntm[row[0]] = row[1].values / sum(row[1].values)

    # read clean datasets
    train_sentences = read_conll(train_clean_filename)
    dev_sentences = read_conll(dev_clean_filename)
    all_sentences = read_conll(all_clean_filename)

    # create simulated noisy train and dev sets
    new_train = add_label_noise(train_sentences,labels=labels,noise_transition_matrix =ntm)
    new_dev = add_label_noise(dev_sentences,labels=labels,noise_transition_matrix =ntm)

    # save train and dev sets 
    save_to_column_file(simulated_data_path+corpus_name+'.train',new_train)
    save_to_column_file(simulated_data_path+corpus_name+'.dev',new_dev)

    # merge simulated train and dev sets into .traindev (maintaining original sentence order)
    dev_ind = 1
    train_ind = 1
    new_sentences = []
    new_sentences.append(all_sentences[0])
    for s in all_sentences[1:]:
        sentence_text = ''.join([t[0] for t in s])
        if train_ind < len(new_train) and sentence_text == ''.join([t[0] for t in new_train[train_ind]]):
            new_sentences.append(new_train[train_ind])
            train_ind += 1
        elif dev_ind < len(new_dev) and sentence_text == ''.join([t[0] for t in new_dev[dev_ind]]):
            new_sentences.append(new_dev[dev_ind])
            dev_ind += 1
        else:
            if dev_ind < len(new_dev):
                input('wait')
            else:
                if 'DOCSTART' in s[0][0]:
                    new_sentences.append(s)

    # save merged dataset
    save_to_column_file(simulated_data_path+corpus_name+'.traindev',new_sentences)

os.system(f'cp {noisebench_data_path}clean.test {simulated_data_path}clean.test')
