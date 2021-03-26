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

# fig, axes = plt.subplots(1, 3, figsize=(15, 8))

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

######################################################
#feat = pd.concat([demo, mri], axis=1)
#feat = feat.drop(['APOEe4'], axis=1)
#feat = (feat - feat.min()) / (feat.max() - feat.min())

#X = feat.to_numpy()
# X = scaler.fit_transform(np.c_[emb, features_map]) # transform to array
# X = scaler.fit_transform(features_map) # transform to array
# X = scaler.fit_transform(emb[~allzeros])
#y = labels
# y = labels[~allzeros]





# plot embedding
#import numpy as np

#emb = np.loadtxt('../data/adni/emb_corr_coef/n2v_1_1.emb', delimiter=' ', skiprows=1)
#nodes = np.array(emb[:,0], dtype=np.int32)
#emb = np.delete(emb, 0, axis=1)

#emb.shape

#idx_map = {j:i for i, j in enumerate(idx)}
#labels_map = labels[np.array(list(map(idx_map.get, nodes)), dtype=np.int32)]

#plot_embedding(emb, labels_map, 30)

#features_map = features.numpy()
#features_map = features_map[np.array(list(map(idx_map.get, nodes)), dtype=np.int32),:]

#plot_embedding(np.c_[emb, features_map], labels_map, 30)
