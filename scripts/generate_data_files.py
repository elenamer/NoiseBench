from utils import *
import os


SAVE_TRAINDEV_FILE = False


def create_train_dev_splits(filename, all_sentences = None, datestring = '1996-08-24'):
    if not all_sentences:
        all_sentences = read_conll(filename)

    train_sentences = [] 
    dev_sentences = [] 
    for i, s in enumerate(all_sentences):
        if 'DOCSTART' in s[0][0]:
            assert i+3 < len(all_sentences) # last document is too short
            
            # news date is usually in 3rd or 4th sentence of each article
            if datestring in all_sentences[i+2][-1][0] or datestring in all_sentences[i+3][-1][0]:
                save_to_dev = True
            else:
                save_to_dev = False

        if save_to_dev:
            dev_sentences.append(s)
        else:
            train_sentences.append(s)

    save_to_column_file(os.sep.join(filename.split(os.sep)[:-1])+os.sep+filename.split(os.sep)[-1].split('.')[0]+'.dev',dev_sentences)
    save_to_column_file(os.sep.join(filename.split(os.sep)[:-1])+os.sep+filename.split(os.sep)[-1].split('.')[0]+'.train',train_sentences)


def merge_tokens_labels(corpus, all_clean_sentences, token_indices):
    # generate NoiseBench dataset variants, given CleanCoNLL, noisy label files and index file

    noisy_labels = read_conll(os.path.join('data','annotations',f'{corpus}.traindev'))

    for index, sentence in zip(token_indices, noisy_labels):

        if index.strip() == 'docstart':
            assert len(sentence) == 1
            sentence[0][0] = '-DOCSTART-'
            continue

        clean_sentence = all_clean_sentences[int(index.strip())]

        assert len(clean_sentence) == len(sentence) # this means indexing is wrong

        for token, label in zip(clean_sentence, sentence):
            label[0] = token[0] # token[0] -> text, token[1] -> BIO label
    if SAVE_TRAINDEV_FILE:
        save_to_column_file(os.path.join('data','noisebench',f'{corpus}.traindev'),noisy_labels)
    return noisy_labels


def main():

    index_file = open(os.path.join('data','annotations','index.txt'))
    token_indices = index_file.readlines()

    all_clean_sentences = read_conll(os.path.join('data','cleanconll','cleanconll.train'))

    os.makedirs(os.path.join('data','noisebench'), exist_ok=True)

    for corpus in ['clean', 'noise_expert','noise_crowd','noise_crowdbest','noise_distant','noise_weak','noise_llm']:

        noisy_sentences = merge_tokens_labels(corpus, all_clean_sentences, token_indices)
        create_train_dev_splits(all_sentences=noisy_sentences,filename=os.path.join('data','noisebench',f'{corpus}.traindev'))
    
    # copy test set 
    all_clean_test_sentences = read_conll(os.path.join('data','cleanconll','cleanconll.test'))
    
    test_sentences = []
    for s in all_clean_test_sentences:
        new_s = []
        for token in s:
            new_s.append([token[0],token[4]])
        test_sentences.append(new_s)

    save_to_column_file(os.path.join('data','noisebench',f'clean.test'),test_sentences)

if __name__ == "__main__":
    main()