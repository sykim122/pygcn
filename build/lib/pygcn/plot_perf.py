import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from utils import *

dataset = 'adni'

# CFs = ['SVC', 'RFC', 'LR-lasso', 'LR-ridge', 'gaussNB', 'MLP', "GB"]

#CFs = ['SVC', 'RFC', 'LR-ridge', 'MLP']

#f, axes = plt.subplots(1, 2, figsize=(10, 8))
#x=0
#for feats in ['dm', 'dma']:
#    __, features, labels, idx = load_data(dataset=dataset, feats=feats, graph=sep, rand_cut=i)

#    dict = classification_test(X=features.numpy(), y=labels.numpy(), CFs=CFs)

#    for sep in ['random', 'corr_coef_pval']:
#    with open('../data/{}/{}_{}_{}.bal.pkl'.format(dataset, dataset, feats, sep), 'rb') as f:
#        tmp = pickle.load(f)
      # print(tmp[len(tmp)-1])
#        dict[i] = tmp[len(tmp)-1]

#    print([(k, np.mean(dict[k]), np.std(dict[k])) for k in dict])

#    sns.boxplot(data=pd.DataFrame(dict), ax=axes[x], palette='Set2')
#    axes[x].set_title(head)
#    axes[x].set_ylim(0, 1.1)
#    x += 1

# plot gcn performance with varying sparsity

feats = 'dma'
sep = 'random'

dict = {}
for i in np.arange(0.1, 1.1, 0.1):
    with open('../data/{}/{}_{}_{}_{:.1f}.bal.pkl'.format(dataset, dataset, feats, sep, i), 'rb') as f:
        tmp = pickle.load(f)
        dict[i] = tmp[len(tmp)-1]

print([(k, np.mean(dict[k]), np.std(dict[k])) for k in dict], sep='\n')

df = pd.DataFrame.from_dict(dict, orient='index')
df = df.stack().reset_index(level=1, drop=True).to_frame(name='AUC')
df['sparsity'] = df.index

print(df.head())

p = sns.lineplot(data=df, x='sparsity', y='AUC')
p.set(ylim=(0.5, 1))
p.figure.savefig('../results/dma_random_sparsity.png')

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
