import numpy as np
import itertools
from typing import Set
from flair.data import Dictionary, Corpus
from flair.datasets import CONLL_03, ColumnCorpus
import sklearn
from collections import Counter
import os

def read_conll(filename):
    raw = open(filename, 'r').readlines()
    all_x = []
    point = []
    for line in raw:
        if '\t' in line.strip():
            stripped_line = line.strip().split('\t')
        else:
            stripped_line = line.strip().split(' ')
        point.append(stripped_line)
        if line == '\n':
            if len(point[:-1]) > 0:
                all_x.append(point[:-1])
            point = []
    all_x = all_x
    return all_x

def save_to_column_file(filename, list):
    with open(filename, "w") as f:
        for sentence in list:
            for token in sentence:
                f.write('\t'.join(token))
                f.write('\n')
            f.write('\n')

def calculate_entity_metrics(path,gold_label_file, label_file, ind_noisy_column = 1):
    sentence_id=0
    exclude_labels = []
    all_spans: Set[str] = set()
    all_true_values = {}
    all_predicted_values = {}
    tag_dictionary_gt = None

    corpus_gt: Corpus = ColumnCorpus(data_folder=path, train_file = gold_label_file, sample_missing_splits=False, column_format={0:'text', 1: 'gt'}, document_separator_token="-DOCSTART-",in_memory=False)
    corpus: Corpus = ColumnCorpus(data_folder=path, train_file = label_file, sample_missing_splits=False, column_format={0:'text', ind_noisy_column: 'noisy'}, document_separator_token="-DOCSTART-",in_memory=False)
    gold_label_type = 'gt'
    label_type = 'noisy'

    # count entities
    num_entities = 0 # number of entities in noisy
    num_entities_clean = 0 # number of entities in clean 
    full_matched_clean=0 # correct entities (both boundary and type)

    # types of errors 
    missing = 0
    wrong_boundary = 0
    wrong_type = 0

    # partial matches
    true_partial_matches_clean = 0 # clean is the longer
    true_partial_matches_noisy = 0 # noisy is the longer
    extended_partial_matches = 0 # count only overlapping boundaries, not subsets
    for datapoint, datapoint_gt in zip(corpus.train,corpus_gt.train):
        if 'DOCSTART' in datapoint.text:
            continue
        for gold_label in datapoint_gt.get_labels(gold_label_type):
            representation = str(sentence_id) + ": " + gold_label.unlabeled_identifier
            value = gold_label.value

            if representation not in all_true_values:
                all_true_values[representation] = [value]
            else:
                all_true_values[representation].append(value)

            if representation not in all_spans:
                all_spans.add(representation)

            ## count partial matches (clean span is longer)
            start = gold_label.data_point.tokens[0].idx-1 
            end = gold_label.data_point.tokens[-1].idx-1

            for predicted_span in datapoint.get_labels(label_type):
                start_predicted = predicted_span.data_point.tokens[0].idx-1
                end_predicted = predicted_span.data_point.tokens[-1].idx-1
                if start_predicted >= start and end_predicted<= end:
                    if (end_predicted-start_predicted != end - start) and (gold_label.value == predicted_span.value):
                        true_partial_matches_clean+=1 #
                    elif (gold_label.value == predicted_span.value):
                        full_matched_clean+=1
                if (start_predicted>=start and end_predicted>=end and start_predicted<end) or (start_predicted<=start and end_predicted<=end and end_predicted>start):
                    if (end_predicted-start_predicted != end - start)  and (gold_label.value == predicted_span.value): #same type, but not exactly the same boundary
                        extended_partial_matches+=1 

        for predicted_span in datapoint.get_labels(label_type):
            representation = str(sentence_id) + ": " + predicted_span.unlabeled_identifier

            # add to all_predicted_values
            if representation not in all_predicted_values:
                all_predicted_values[representation] = [predicted_span.value]
            else:
                all_predicted_values[representation].append(predicted_span.value)

            if representation not in all_spans:
                all_spans.add(representation)

            #count partial matches (noisy is longer)
            start = predicted_span.data_point.tokens[0].idx-1
            end = predicted_span.data_point.tokens[-1].idx-1

            for gold_label in datapoint_gt.get_labels(gold_label_type):
                start_true = gold_label.data_point.tokens[0].idx-1
                end_true = gold_label.data_point.tokens[-1].idx-1
                #print(start)
                #print(start_true)
                if start_true >= start and end_true<= end:
                    if (end_true-start_true != end - start) and (gold_label.value == predicted_span.value):
                        true_partial_matches_noisy+=1
                if (start_true>=start and end_true>=end and start_true<end) or (start_true<=start and end_true<=end and end_true>start):
                    if (end_true-start_true != end - start) and (gold_label.value == predicted_span.value):
                        extended_partial_matches+=1

        num_entities += len(datapoint.get_labels(label_type))
        num_entities_clean += len(datapoint_gt.get_labels(gold_label_type))
        sentence_id += 1

    # convert true and predicted values to two span-aligned lists
    true_values_span_aligned = []
    predicted_values_span_aligned = []
    for span in all_spans:
        list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else ["O"]
        # delete exluded labels if exclude_labels is given
        for excluded_label in exclude_labels:
            if excluded_label in list_of_gold_values_for_span:
                list_of_gold_values_for_span.remove(excluded_label)
        # if after excluding labels, no label is left, ignore the datapoint
        if not list_of_gold_values_for_span:
            continue
        true_values_span_aligned.append(list_of_gold_values_for_span)
        predicted_values_span_aligned.append(
            all_predicted_values[span] if span in all_predicted_values else ["O"]
        )

    # make the evaluation dictionary
    evaluation_label_dictionary = Dictionary(add_unk=False)
    evaluation_label_dictionary.add_item("O")
    for true_values in all_true_values.values():
        for label in true_values:
            evaluation_label_dictionary.add_item(label)
    for predicted_values in all_predicted_values.values():
        for label in predicted_values:
            evaluation_label_dictionary.add_item(label)
    print(evaluation_label_dictionary)

    # check if this is a multi-label problem
    # and count types of errors
    multi_label = False
    for true_instance, predicted_instance in zip(true_values_span_aligned, predicted_values_span_aligned):
        if len(true_instance) > 1 or len(predicted_instance) > 1:
            multi_label = True
            break

        if true_instance != ['O'] and predicted_instance == ['O']:
            missing+=1
        if true_instance == ['O'] and predicted_instance != ['O']:
            wrong_boundary+=1
        if true_instance != ['O'] and predicted_instance != ['O'] and predicted_instance!=true_instance:
            wrong_type+=1

    # compute numbers by formatting true and predicted such that Scikit-Learn can use them
    y_true = []
    y_pred = []
    if multi_label:
        # multi-label problems require a multi-hot vector for each true and predicted label
        for true_instance in true_values_span_aligned:
            y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
            for true_value in true_instance:
                y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
            y_true.append(y_true_instance.tolist())

        for predicted_values in predicted_values_span_aligned:
            y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
            for predicted_value in predicted_values:
                y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
            y_pred.append(y_pred_instance.tolist())
    else:
        # single-label problems can do with a single index for each true and predicted label
        y_true = [
            evaluation_label_dictionary.get_idx_for_item(true_instance[0])
            for true_instance in true_values_span_aligned
        ]
        y_pred = [
            evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
            for predicted_instance in predicted_values_span_aligned
        ]

    # now, calculate evaluation numbers
    target_names = []
    labels = []

    counter = Counter(itertools.chain.from_iterable(all_true_values.values()))
    counter.update(list(itertools.chain.from_iterable(all_predicted_values.values())))
    label_names = []
    for label_name, count in counter.most_common():
        if label_name == "O":
            continue
        target_names.append(label_name)
        labels.append(evaluation_label_dictionary.get_idx_for_item(label_name))
        label_names.append(label_name)
    # there is at least one gold label or one prediction (default)
    if len(all_true_values) + len(all_predicted_values) > 1:
        classification_report = sklearn.metrics.classification_report(
            y_true,
            y_pred,
            digits=4,
            target_names=target_names,
            zero_division=0,
            labels=labels,
        )
        print(classification_report)
        classification_report_dict = sklearn.metrics.classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
            labels=labels,
        )
        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred,labels=labels)
        micro_metric =classification_report_dict['micro avg']['f1-score'] if 'micro avg' in classification_report_dict else classification_report_dict['accuracy']
        
        return {'f1_macro':classification_report_dict['macro avg']['f1-score'], 
                'f1':micro_metric , 
                'conf_mat':conf_mat, 
                'labels':label_names, 
                'prec':classification_report_dict['micro avg']['precision'],
                'rec':classification_report_dict['micro avg']['recall'],
                'errors_missing':missing,
                'errors_wrong':wrong_boundary,
                'errors_wrong_type':wrong_type,
                'num_entities':num_entities,
                'num_entities_clean':num_entities_clean,
                'num_correct_entities':full_matched_clean,
                'clean_partial_matches':true_partial_matches_clean,
                'noisy_partial_matches':true_partial_matches_noisy,
                'errors_partial_matches':true_partial_matches_noisy + true_partial_matches_clean,
                'extended_partial_matches':extended_partial_matches,
                'dict_report':classification_report_dict
                }

def calculate_token_f1(path,gold_label_file, label_file):
    sentences = read_conll(path+os.sep+label_file)
    sentences_clean = read_conll(path+os.sep+gold_label_file)
    all_true = []
    all_pred = []
    for s_clean, s in zip(sentences_clean, sentences):
        for t_clean, t in zip(s_clean, s):
            all_true.append(t_clean[1])
            all_pred.append(t[1])
    labels = list(set(all_true))
    conf_mat = sklearn.metrics.confusion_matrix(all_true,all_pred, labels = labels)
    return {'macro_f1':sklearn.metrics.f1_score(all_true, all_pred, average='macro'), 'micro_f1':sklearn.metrics.f1_score(all_true, all_pred, average='micro'),'conf_mat':conf_mat,'labels':labels}

def calculate_sentence_accuracy(path,gold_label_file, label_file):
    corpus_gt: Corpus = ColumnCorpus(data_folder=path, train_file = gold_label_file, sample_missing_splits = False,column_format={0:'text', 1: 'gt'}, document_separator_token="-DOCSTART-", in_memory=False)
    corpus: Corpus = ColumnCorpus(data_folder=path, train_file = label_file,sample_missing_splits = False,column_format={0:'text', 1: 'noisy'}, document_separator_token="-DOCSTART-",in_memory=False)
    count_correct = 0
    count_incorrect = 0
    for datapoint, datapoint_gt in zip(corpus.train,corpus_gt.train):
        if str(datapoint) != str(datapoint_gt):
            count_incorrect += 1
        else:
            count_correct += 1
    return count_correct/(count_correct+count_incorrect)


def find_seen_entities(path, gold_label_file, label_file):
    all_entities = {}
    print(label_file)
    print(gold_label_file)
    corpus_gt: Corpus = ColumnCorpus(data_folder=path, train_file = gold_label_file, sample_missing_splits = False, column_format={0:'text', 1: 'gt'}, document_separator_token="-DOCSTART-",in_memory=False)
    corpus: Corpus = ColumnCorpus(data_folder=path, train_file = label_file,sample_missing_splits = False,column_format={0:'text', 1: 'noisy'}, document_separator_token="-DOCSTART-",in_memory=False)

    gold_label_type = 'gt'
    label_type = 'noisy'

    for datapoint, datapoint_gt in zip(corpus.train,corpus_gt.train):

        if 'DOCSTART' in datapoint.text:
            continue

        for gold_label in datapoint_gt.get_labels(gold_label_type):
            representation = gold_label.data_point.text
            value = gold_label.value
            if representation not in all_entities:
                all_entities[representation]={'train_noisy':set(), 'train_clean':set()}
            all_entities[representation]['train_clean'].add(value)

        for noisy_label in datapoint.get_labels(label_type):
            representation = noisy_label.data_point.text
            value = noisy_label.value
            if representation not in all_entities:
                all_entities[representation]={'train_noisy':set(), 'train_clean':set()}
            all_entities[representation]['train_noisy'].add(value)

    # all_entities[string]['train_noisy'] : set of all different type labels found for the same surface form in the noisy label set
    # all_entities[string]['train_clean'] : set of all different type labels found for the same surface form in the clean label set

    seen_clean = []
    seen_noisy = []

    for surface_form in all_entities.keys():

        is_clean = False

        noisy_mentions = all_entities[surface_form]['train_noisy']
        clean_mentions = all_entities[surface_form]['train_clean']

        count_appearances_clean = len(list(noisy_mentions)) 
        count_appearances_noisy = len(list(clean_mentions))
        
        if count_appearances_clean >0 and count_appearances_noisy >0:
            # seen both in clean and noisy label set, with the same type
            for entity_type in noisy_mentions:
                for entity_type_clean in clean_mentions:
                    if entity_type == entity_type_clean:
                        is_clean = True
                        seen_clean.append(surface_form)
            
        # if count_appearances_clean >0 and count_appearances_noisy >0 -> seen with wrong type label
        # if count_appearances_clean >0 and count_appearances_noisy == 0 -> this mention is missing in the noisy labels, however the same string has been seen by the model
        # if count_appearances_clean == 0 and count_appearances_noisy > 0 -> this string is a false positive mention in the noisy labels
        if count_appearances_clean >0 or count_appearances_noisy >0:
            # surface form seen with a wrong label
            if not is_clean:
                seen_noisy.append(surface_form)

    return {'seen_clean':seen_clean,'seen_noisy':seen_noisy}


def calculate_entity_f1(path,gold_label_file, label_file, seen_entities, ind_noisy_column=1, evaluate_on_seen=False): # ind_noisy_column or ind_predicted_column
    sentence_id=0
    exclude_labels = []
    all_spans: Set[str] = set()
    all_true_values = {}
    all_predicted_values = {}
    all_predicted_values_all = {}

    corpus_gt: Corpus = ColumnCorpus(data_folder=path, train_file = gold_label_file,sample_missing_splits = False,column_format={0:'text', 1: 'gt'}, document_separator_token="-DOCSTART-",in_memory=False)
    corpus: Corpus = ColumnCorpus(data_folder=path, train_file = label_file,sample_missing_splits = False,column_format={0:'text', ind_noisy_column: 'noisy'}, document_separator_token="-DOCSTART-",in_memory=False)
    gold_label_type = 'gt'
    label_type = 'noisy'

    for datapoint, datapoint_gt in zip(corpus.train,corpus_gt.train):
        if 'DOCSTART' in datapoint.text:
            continue
        excluded_spans = []
        included_spans = []

        for gold_label in datapoint_gt.get_labels(gold_label_type):
            text = gold_label.data_point.text
            representation = str(sentence_id) + ": " + gold_label.unlabeled_identifier
            value = gold_label.value

            # mark all excluded and included tokens (as a list of tuples)
            # all included are taken into account
            # all excluded are excluded
            # (if evaluate on seen)
            # out of the remaining ones, if they are found in a predicted entity, include them only then, otherwise exclude them
            # (if evaluate on unseen)
            # out of the remaining ones, if they are found in a predicted entity, exclude them only then, otherwise include them

            if (evaluate_on_seen and text in seen_entities) or (not evaluate_on_seen and text not in seen_entities):
                if representation not in all_true_values:
                    all_true_values[representation] = [value]
                else:
                    all_true_values[representation].append(value)

                if representation not in all_spans:
                    all_spans.add(representation)
                included_spans.append((gold_label.data_point.tokens[0].idx-1, gold_label.data_point.tokens[-1].idx-1)) # each span is a tuple
            else:
                excluded_spans.append((gold_label.data_point.tokens[0].idx-1, gold_label.data_point.tokens[-1].idx-1)) # each span is a tuple

        # set indices of both included and excluded spans
                
        span_id = 0 # index of excluded span (within list of excluded spans)
        span_id_existing = 0 # index of existing span (within list of existing spans)

        if len(excluded_spans)>0:
            start_true = excluded_spans[0][0]
            end_true = excluded_spans[0][1]
        else:
            # move to the end
            start_true = len(datapoint.tokens)
            end_true = len(datapoint.tokens)

        if len(included_spans)>0:
            start_true_existing = included_spans[0][0]
            end_true_existing = included_spans[0][1]
        else:
            # move to the end
            start_true_existing = len(datapoint.tokens)
            end_true_existing = len(datapoint.tokens)

        for predicted_span in datapoint.get_labels(label_type):

            representation = str(sentence_id) + ": " + predicted_span.unlabeled_identifier
            text = predicted_span.data_point.text

            # find the span of interest within the list of excluded 
            while predicted_span.data_point.tokens[0].idx-1 > end_true:
                span_id += 1
                if span_id >= len(excluded_spans):
                    start_true = len(datapoint.tokens) +1
                    end_true = len(datapoint.tokens)+1
                else:
                    # move to the next excluded tuple
                    start_true = excluded_spans[span_id][0]
                    end_true = excluded_spans[span_id][1]
            
            # find the span of interest within the list of included 
            while predicted_span.data_point.tokens[0].idx-1 > end_true_existing:
                span_id_existing += 1

                if span_id_existing >= len(included_spans):
                    start_true_existing = len(datapoint.tokens)
                    end_true_existing = len(datapoint.tokens)
                else:
                    # move to the next included tuple
                    start_true_existing = included_spans[span_id_existing][0]
                    end_true_existing = included_spans[span_id_existing][1]
            
            is_within_an_excluded_span = (predicted_span.data_point.tokens[0].idx-1 >= start_true) and (predicted_span.data_point.tokens[-1].idx-1 <= end_true)
            does_cover_an_excluded_span = (predicted_span.data_point.tokens[0].idx-1 <= start_true) and (predicted_span.data_point.tokens[-1].idx-1 >= end_true)
            
            # if entity doesn't match an excluded token
            if not (is_within_an_excluded_span or does_cover_an_excluded_span): 

                is_within_an_included_span = (predicted_span.data_point.tokens[0].idx-1 >= start_true_existing) and (predicted_span.data_point.tokens[-1].idx-1 <= end_true_existing)
                does_cover_an_included_span = (predicted_span.data_point.tokens[0].idx-1 <= start_true_existing) and (predicted_span.data_point.tokens[-1].idx-1 >= end_true_existing)
                
                # if we evaluate on unseen or entity matches an included token
                if (not evaluate_on_seen) or (does_cover_an_included_span or is_within_an_included_span): 
                    if representation not in all_predicted_values:
                        all_predicted_values[representation] = [predicted_span.value]
                    else:
                        all_predicted_values[representation].append(predicted_span.value)

                    if representation not in all_spans:
                        all_spans.add(representation)
                        
            if representation not in all_predicted_values_all:
                all_predicted_values_all[representation] = [predicted_span.value]
            else:
                all_predicted_values_all[representation].append(predicted_span.value)
        sentence_id += 1

    # convert true and predicted values to two span-aligned lists
    true_values_span_aligned = []
    predicted_values_span_aligned = []
    for span in all_spans:
        list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else ["O"]
        # delete exluded labels if exclude_labels is given
        for excluded_label in exclude_labels:
            if excluded_label in list_of_gold_values_for_span:
                list_of_gold_values_for_span.remove(excluded_label)
        # if after excluding labels, no label is left, ignore the datapoint
        if not list_of_gold_values_for_span:
            continue
        true_values_span_aligned.append(list_of_gold_values_for_span)
        predicted_values_span_aligned.append(
            all_predicted_values[span] if span in all_predicted_values else ["O"]
        )

    # make the evaluation dictionary
    evaluation_label_dictionary = Dictionary(add_unk=False)
    evaluation_label_dictionary.add_item("O")

    for true_values in all_true_values.values():
        for label in true_values:
            evaluation_label_dictionary.add_item(label)
    for predicted_values in all_predicted_values.values():
        for label in predicted_values:
            evaluation_label_dictionary.add_item(label)

    print(evaluation_label_dictionary)

    # compute numbers by formatting true and predicted such that Scikit-Learn can use them
    y_true = []
    y_pred = []

    # single-label problems can do with a single index for each true and predicted label
    y_true = [
        evaluation_label_dictionary.get_idx_for_item(true_instance[0])
        for true_instance in true_values_span_aligned
    ]
    y_pred = [
        evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
        for predicted_instance in predicted_values_span_aligned
    ]

    # now, calculate evaluation numbers
    counter = Counter(itertools.chain.from_iterable(all_true_values.values()))
    counter.update(list(itertools.chain.from_iterable(all_predicted_values.values())))
    
    labels = []
    label_names = []
    
    for label_name, count in counter.most_common():
        if label_name == "O":
            continue
        labels.append(evaluation_label_dictionary.get_idx_for_item(label_name))
        label_names.append(label_name)

    if len(all_true_values) + len(all_predicted_values) > 1:
        
        classification_report_dict = sklearn.metrics.classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
            labels=labels,
        )

        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred,labels=[0]+labels)

        micro_metric =classification_report_dict['micro avg']['f1-score'] if 'micro avg' in classification_report_dict else classification_report_dict['accuracy']
        
        return {'f1':micro_metric , 
                'conf_mat':conf_mat, 
                'labels':label_names, }
