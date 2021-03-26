dataset = 'adni'
feats = 'dma'
sep = 'corr_coef_pval'

import numpy as np
from utils import *

#pval_cut = 5e-10
#for i in np.arange(1,11,1):
#    print(pval_cut)
adj, features, labels, idx = load_data(dataset=dataset, feats=feats, graph=sep, pval_cut=-1)
    
#    pval_cut = pval_cut * 10

