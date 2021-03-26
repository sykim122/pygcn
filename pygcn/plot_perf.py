import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from utils import *

dataset = 'smc'
feats = 'dma'

# CFs = ['SVC', 'RFC', 'LR-lasso', 'LR-ridge', 'gaussNB', 'MLP', "GB"]
CFs = ['SVC', 'RFC', 'LR-ridge', 'MLP']

group = ['NC', 'MCI', 'NC+MCI']

df_concat = pd.DataFrame(columns = ['model', 'AUC'])
x=0
for group_sep in ['.nc', '.mci', '']:

    __, features, labels, idx = load_data(dataset=dataset, feats=feats)

    d = classification_test(X=features.numpy(), y=labels.numpy(), CFs=CFs)

    for sep in ['random', 'corr_coef_pval']:
        if(sep == 'random'):
            sp = '1.0'
        else:
            sp = 'top1.0'

        with open('../results/{}/{}_{}_{}_{}{}.bal.pkl'.format(dataset, dataset, feats, sep, sp, group_sep), 'rb') as f:
            tmp = pickle.load(f)
            # print(tmp[len(tmp)-1])
            d[sep] = tmp[len(tmp)-1]

    df = pd.DataFrame(d).stack().reset_index()
    df = df.iloc[:,1:]
    df.columns = ['model', 'AUC']
    df.insert(0, 'group', group[x])

    df_concat = pd.concat([df_concat, df], axis=0)

    x += 1

df_concat.to_csv('../results/adni_dma_perf.csv')
print(df_concat.groupby(['group', 'model']).agg(['mean', 'std']))

p = sns.catplot(
        data=df_concat, x='model', y='AUC',
        col='group', kind='box', palette='Set2')

p.tight_layout()
p.savefig('../results/smc_dma_perf.png')


