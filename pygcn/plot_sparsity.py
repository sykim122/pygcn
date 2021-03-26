import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from utils import *


def get_results(dataset='adni', feats='dma', sep='random'):
    dict = {}
    for i in np.concatenate([[1.0], np.arange(10, 101, 10)]):
        if(sep == 'random'):
            sp = '{:.1f}'.format(i)
        else:
            sp = 'top{:.1f}'.format(i)

        with open('../results/{}/{}_{}_{}_{}.bal.pkl'.format(dataset, dataset, feats, sep, sp), 'rb') as f:
            tmp = pickle.load(f)
            dict[i] = tmp[len(tmp)-1]

    df = pd.DataFrame.from_dict(dict, orient='index')
    df = df.stack().reset_index(level=1, drop=True).to_frame(name='AUC')
    # df['connectivity'] = df.index
    df['connectivity'] = ["%g" % number for number in df.index]
    df.insert(0, 'graph', sep)
    # print(df.head(10))
  
    return df

df_random = get_results(dataset='smc', sep='random')
df_corr = get_results(dataset='smc', sep='corr_coef_pval')

df = pd.concat([df_corr, df_random], axis=0)

print(df.groupby(['graph', 'connectivity']).agg(['mean', 'std']))

plt.figure(figsize=(6, 4), dpi=100)

p = sns.lineplot(
        data=df,
        x="connectivity", y="AUC", hue="graph", style="graph",
        markers=True, dashes=False, palette='Set1',
    )
p.figure.savefig('../results/smc_dma_sparsity.png')
