import flair
import torch
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
import os
import argparse
import numpy as np
from collections import defaultdict
import json

def main():

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-c","--config", help="filename with experiment configuration")
    argParser.add_argument("-g", "--gpu", help="set gpu id", default=0)
    # set gpu ID

    args = argParser.parse_args()

    with open(args.config) as json_file:
        config = json.load(json_file)

    print(config)
    flair.device = torch.device('cuda:'+str(args.gpu)) 

    data_path = config['paths']['data_path']
    experiment_path = config['paths']['resources_path']
    corpora = config['corpora']
    train_extension = config['paths']['train_filename_extension']
    dev_extension = config['paths']['dev_filename_extension']
    test_extension = config['paths']['test_filename_extension']
    learning_rates = config['parameters']['learning_rate']
    batch_sizes = config['parameters']['batch_size']
    num_epochs = config['parameters']['num_epochs']

    seeds = [100, 500, 13]
    tag_type = 'ner'

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    
    with open(experiment_path+'config.json', 'w') as f:
        json.dump(config, f)

    for corpus_name in corpora:

        train_filename = f'{data_path}{corpus_name}{train_extension}'

        if 'clean' in dev_extension:
            dev_filename = f'{data_path}{dev_extension}'
        else:
            dev_filename = f'{data_path}{corpus_name}{dev_extension}'

        if 'clean' in test_extension:
            test_filename = f'{data_path}{test_extension}'
        else:
            test_filename = f'{data_path}{corpus_name}{test_extension}'

        f1_scores=defaultdict(list)
        dev_f1_scores=defaultdict(list)
        output_path=config['paths']['resources_path']+corpus_name

        for bs in batch_sizes:
            for lr in learning_rates:
                label = str(bs)+'_'+str(lr)

                conll_corpus = ColumnCorpus(data_folder='./',
                                            column_format={0: "text", 1: tag_type},
                                            document_separator_token="-DOCSTART-",
                                            train_file=train_filename, 
                                            dev_file=dev_filename, 
                                            test_file=test_filename
                                            )

                tag_dictionary = conll_corpus.make_label_dictionary(label_type=tag_type, add_unk=False)

                temp_f1_scores = []
                temp_dev_f1_scores = []

                for seed in seeds:
                    flair.set_seed(seed)

                    embeddings = TransformerWordEmbeddings(model=config['parameters']['model'],
                                                        layers="-1",
                                                        subtoken_pooling="first",
                                                        fine_tune=True,
                                                        use_context=True,
                                                        )

                    tagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type='ner',
                                            use_crf=False,
                                            use_rnn=False,
                                            reproject_embeddings=False,
                                            )

                    trainer = ModelTrainer(tagger, conll_corpus)

                    fine_tuning_args = {
                        'base_path':output_path+os.sep+label+os.sep+str(seed),
                        'learning_rate':float(lr),
                        'mini_batch_size':int(bs),
                        'max_epochs':num_epochs,
                        'save_final_model' : False,
                        'monitor_test' : config['parameters']['monitor_test'],
                        'monitor_train' : config['parameters']['monitor_train'],
                    }

                    if config['parameters']['scheduler'] and config['parameters']['scheduler'] == 'None':
                        fine_tuning_args['scheduler'] = None

                    out = trainer.fine_tune(**fine_tuning_args)
                    
                    temp_f1_scores.append(out['test_score'])

                f1_scores[label] = (np.mean(temp_f1_scores), np.std(temp_f1_scores))

                with open(output_path+os.sep+label+os.sep+'test_results.tsv', 'w') as f:
                    f.write('params\tmean\tstd\n')
                    label = str(bs)+'_'+str(lr)
                    f.write(f'{label} \t{str(f1_scores[label][0])} \t {str(f1_scores[label][1])} \n')

        with open(output_path+os.sep+'test_results.tsv', 'w') as f:
            f.write('params\tmean\tstd\n')
            for bs in batch_sizes:
                for lr in learning_rates:
                    label = str(bs)+'_'+str(lr)
                    f.write(f'{label} \t{str(f1_scores[label][0])} \t {str(f1_scores[label][1])} \n')

if __name__ == "__main__":
    main()