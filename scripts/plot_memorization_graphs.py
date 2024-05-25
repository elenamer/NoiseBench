import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sns.set(style="whitegrid", font_scale=2)

def plot_memorization(ax, data_noisy, data_clean):
    ax.plot(data_noisy, '-',label='train set with noisy labels',linewidth=3)
    ax.plot(data_clean, '--',label='train set with clean labels',linewidth=3)

    ax.set_xticks(list(ax.get_xticks())[1:-1] + [10])
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(0,1)
    ax.axvline(x=10,ls='--', color='gray',ymin=0, ymax=5,linewidth=1.5)


corpora = ['noise_expert','noise_crowdbest','noise_distant']

experiment_path_real_noise = 'resources/exp2_memorization_real_noise'
experiment_path_simulated_noise = 'resources/exp2_memorization_simulated_noise'
experiment_parameters = '32_5e-06'
num_epochs = 100

results_path = 'results/memorization_plots'
memorization_grid_filename = 'memorization_grid.png'

ns_real = {'noise_expert':'5.5%','noise_crowdbest':'15.3%','noise_distant':'31.3%'}
ns_cdn =  {'noise_expert':'5.9%','noise_crowdbest':'17.9%','noise_distant':'39.2%'}

#titles = {'noisy_original':'Expert','noisy_mv_oracle':'Crowd','noisy_mv':'Crowdsourcing majority vote 36% - "noisy_mv"','noisy_bond':'Distant','noisy_wrench':'Weak supervision 40% - "noisy_wrench"','noisy_fabricator':'Fabricator GPT3.5 45% - "noisy_fabricator"'}

full_df = pd.DataFrame(columns=['mean','std'])
fig_grid, ax_grid = plt.subplots(2, 3, figsize=(24, 12), dpi=150)

for i, corpus_name in enumerate(corpora):
    noisy_train = []
    noisy_dev = []
    cdn_train = []
    cdn_dev = []
    f1_scores=[]
    for seed in [100,500,13]:  

        path_to_noisy =  f'{experiment_path_real_noise}/{corpus_name}/{experiment_parameters}/{str(seed)}/loss.tsv'
        path_to_cdn = f'{experiment_path_simulated_noise}/{corpus_name}_cdn/{experiment_parameters}/{str(seed)}/loss.tsv'

        noisy_df =  pd.read_csv(path_to_noisy, delimiter='\t', header=0, index_col=0)
        cdn_df =  pd.read_csv(path_to_cdn, delimiter='\t', header=0, index_col=0)

        noisy_train.append(noisy_df['TRAIN_F1'].values)
        noisy_dev.append(noisy_df['DEV_F1'].values)

        cdn_train.append(cdn_df['TRAIN_F1'].values)
        cdn_dev.append(cdn_df['DEV_F1'].values)

    cdn_dev=np.array(cdn_dev).mean(axis=0)
    cdn_train=np.array(cdn_train).mean(axis=0)

    noisy_dev=np.array(noisy_dev).mean(axis=0)
    noisy_train=np.array(noisy_train).mean(axis=0)

    # plot subplots of grid
    sns.set(style="whitegrid", font_scale=2) #resize the text (larger for memorization plots grid)

    plot_memorization(ax_grid[0,i],noisy_train, noisy_dev )
    plot_memorization(ax_grid[1,i],cdn_train, cdn_dev )

    ax_grid[0, i].set_title("Realistic noise - "+ns_real[corpus_name])
    ax_grid[1, i].set_title("Simulated noise - "+ns_cdn[corpus_name])

    #plot each separately

    sns.set(style="whitegrid", font_scale=1) #resize the text (smaller for separate plots)

    fig_real, ax_real = plt.subplots()
    plot_memorization(ax_real,noisy_train, noisy_dev )
    ax_real.set_xlabel('Epoch')
    ax_real.set_ylabel('F1 score')
    ax_real.legend(loc='lower right')
    fig_real.tight_layout()
    fig_real.savefig(f'{results_path}/plot{str(num_epochs)}_real_{corpus_name}.png', dpi=300)  
    
    fig_simulated, ax_simulated = plt.subplots()
    plot_memorization(ax_simulated,cdn_train, cdn_dev )
    ax_simulated.set_xlabel('Epoch')
    ax_simulated.set_ylabel('F1 score')
    ax_simulated.legend(loc='lower right')
    fig_simulated.tight_layout()
    fig_simulated.savefig(f'{results_path}/plot{str(num_epochs)}_simulated_{corpus_name}.png', dpi=300)           

sns.set(style="whitegrid", font_scale=2)

y_label = fig_grid.supylabel('F1 score')
x_label = fig_grid.supxlabel('Epoch')
fig_grid.tight_layout()
lgd = fig_grid.legend(labels = ['train set with noisy labels', 'train set with clean labels'], loc="lower right", ncol=2)

fig_grid.savefig(f'{results_path}/{memorization_grid_filename}', dpi=300,bbox_extra_artists=(lgd,y_label,x_label), bbox_inches='tight')     