from utils import *
import pandas as pd
import os

def as_percentage_str(metric):
    return str(round(metric,3)*100)

#set paths
base_data_path = './data/'
base_results_path = './results/data_overviews/'

#set flag if it's simulated (class-dependent noise)
is_simulated_noise = False

if is_simulated_noise:
    noisy_datasets_path = base_data_path+'noisebench/simulated/'
    results_path = base_results_path+'noisebench/simulated/'
    file_name_modifier = ''
else:
    noisy_datasets_path = base_data_path+'noisebench/'
    results_path = base_results_path+'noisebench/'
    file_name_modifier = ''

perclass_overview_path = results_path+os.sep+'benchmark_overview_perclass_metrics.csv'
overview_path = results_path+os.sep+'benchmark_overview_metrics.csv'

# pre-defined order of labels in token-level confusion matrices
label_order = ['O','B-LOC','I-LOC','B-MISC','I-MISC','B-ORG','I-ORG','B-PER','I-PER'] #(temp)

# define list of corpora and metrics to output
corpora = ['noise_expert','noise_crowdbest','noise_crowd','noise_distant','noise_weak','noise_llm']
metric_names_list = ['corpus','Noise level','Acc sentence-level','F1 token-level','F1 entity-level','Prec entity-level','Rec entity-level','num_entities','num_correct_entities','errors_missing','errors_wrong','errors_wrong_type','errors_partial_matches']
metrics_list_dict = dict.fromkeys(metric_names_list)
labels = ['LOC','ORG','PER','MISC']
metrics = ['precision','recall','support']

# open both data overview files and write headers
if not os.path.exists(results_path):
    os.makedirs(results_path)

perclass_file = open(perclass_overview_path,'w')
file = open(overview_path,'w')

for m in metric_names_list:
    file.write(f'{m}\t')
file.write('\n')
perclass_file.write('corpus\t')

for l in labels:
    for m in metrics:
        perclass_file.write(f'{l} {m}\t')
perclass_file.write('\n')

for corpus in corpora:

    print(corpus)
    perclass_file.write(corpus+'\t')

    sentence_acc = calculate_sentence_accuracy(noisy_datasets_path, 'clean.train',f'{corpus}{file_name_modifier}.train')
    token_result = calculate_token_f1(noisy_datasets_path, 'clean.train',f'{corpus}{file_name_modifier}.train')
    result = calculate_entity_metrics(noisy_datasets_path, 'clean.train',f'{corpus}{file_name_modifier}.train') #calculate_entity_metrics (f1, prec and rec are micro)

    metrics_list_dict['corpus'] = corpus
    metrics_list_dict['Acc sentence-level'] = as_percentage_str(sentence_acc)
    metrics_list_dict['Noise level'] = as_percentage_str(1-result['f1'])
    metrics_list_dict['F1 token-level'] = as_percentage_str(token_result['micro_f1'])
    metrics_list_dict['F1 entity-level'] = as_percentage_str(result['f1'])
    metrics_list_dict['Prec entity-level'] = as_percentage_str(result['prec'])
    metrics_list_dict['Rec entity-level'] = as_percentage_str(result['rec'])
    metrics_list_dict['num_entities'] = str(result['num_entities'])
    metrics_list_dict['num_correct_entities'] = str(result['num_correct_entities'])
    metrics_list_dict['errors_missing'] = str(result['errors_missing'])
    metrics_list_dict['errors_wrong'] = str(result['errors_wrong'])
    metrics_list_dict['errors_wrong_type'] = str(result['errors_wrong_type'])
    metrics_list_dict['errors_partial_matches'] = str(result['errors_partial_matches'])

    for m in metric_names_list:
        file.write(f'{metrics_list_dict[m]}\t')
    file.write('\n')

    # save perclass metrics
    for l in labels:
        for m in metrics:
            if m!='support':
                perclass_file.write(as_percentage_str(result['dict_report'][l][m])+'\t')
            else:
                perclass_file.write(str(result['dict_report'][l][m])+'\t')
    perclass_file.write('\n')

    #save token-level ntm (used to generate cdn noise)
    df=pd.DataFrame(token_result['conf_mat'])
    df.columns=token_result['labels']
    df.index=token_result['labels']

    # keep pre-defined label order for all datasets
    df = df[label_order]
    df = df.loc[label_order]

    df.to_csv(results_path+os.sep+'NTM_'+corpus+'.csv') #token-level confusion matrix which serves as a noise transition matrix

file.close()
perclass_file.close()