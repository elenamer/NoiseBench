from utils import *
import os

SAVE_TRAINDEV_FILE = False

def create_train_dev_splits(filename, all_sentences = None, dev_share = 0.17, num_documents = 553):
    if not all_sentences:
        all_sentences = read_conll(filename)

    train_sentences = [] 
    dev_sentences = []

    document_count = 0
    save_to_dev = False

    for i, s in enumerate(all_sentences):
        if 'DOCSTART' in s[0][0]:
            document_count += 1

            if document_count >= int( (1 - dev_share) * num_documents):
                save_to_dev = True

        if save_to_dev:
            dev_sentences.append(s)
        else:
            train_sentences.append(s)

    save_to_column_file(os.sep.join(filename.split(os.sep)[:-1])+os.sep+filename.split(os.sep)[-1].split('.')[0]+'.dev',dev_sentences)
    save_to_column_file(os.sep.join(filename.split(os.sep)[:-1])+os.sep+filename.split(os.sep)[-1].split('.')[0]+'.train',train_sentences)

def main():

    corpora = ['clean','noise_expert']

    os.makedirs(os.path.join('data','noisebench_german'), exist_ok=True)

    #create train and dev files
    only_train = read_conll('data/conll_german/deu.train')

    for corpus in corpora:
        noisy_sentences = []

        all_annotations = read_conll(f'data/annotations_german/{corpus}.deu')

        for sentence, only_labels in zip(only_train, all_annotations[:len(only_train)]):
            assert len(sentence) == len(only_labels)
            new_sentence = []

            for token, labels in zip(sentence, only_labels):
                if token[0] != '-DOCSTART-':
                    assert token[2] == labels[1]

                new_sentence.append([token[0], labels[3]])

            noisy_sentences.append(new_sentence)
            
        if SAVE_TRAINDEV_FILE:
            save_to_column_file(os.path.join('data','noisebench_german',f'{corpus}.traindev'),noisy_sentences)

        create_train_dev_splits(all_sentences=noisy_sentences,filename=os.path.join('data','noisebench_german',f'{corpus}.traindev'))

        
    # copy test set
    only_test = read_conll('data/conll_german/deu.testb')
    new_dataset = []
    for sentence in only_test:   
        new_sentence = []
        for token in sentence:
            new_sentence.append([token[0], token[4]])
        new_dataset.append(new_sentence)
    save_to_column_file('data/noisebench_german/clean.test', new_dataset)

if __name__ == "__main__":
    main()