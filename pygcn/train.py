from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import GCN

from sklearn.model_selection import *

import matplotlib.pyplot as plt



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=550,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-10,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--group', type=str, default='',
                    help='.mci / .nc')
parser.add_argument('--dataset', type=str, default='adni')
parser.add_argument('--feats', type=str, default='dma')
parser.add_argument('--sep', type=str, default='corr_coef_pval',
                    help='random / corr_coef_pval')
parser.add_argument('--rand_cut', type=float, default='0.1')
parser.add_argument('--pval_cut', type=float, default='0.05')
parser.add_argument('--coef_cut', type=float, default='0.7')

parser.add_argument('--top_edges_p', type=float, default='100')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx = load_data(dataset=args.dataset, feats=args.feats, graph=args.sep, group=args.group,
                                    rand_cut=args.rand_cut, pval_cut=args.pval_cut, coef_cut=args.coef_cut, top_edges_p=args.top_edges_p)


# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(input=output[idx_train], target=labels[idx_train], weight=list(labels[idx_train].shape)[0]/(2*torch.bincount(labels[idx_train])))
    acc_train, prec_train, recall_train, fone_train, ap_train, auc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(input=output[idx_val], target=labels[idx_val], weight=list(labels[idx_val].shape)[0]/(2*torch.bincount(labels[idx_val])))
    acc_val, prec_val, recall_val, fone_val, ap_val, auc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'auc_train: {:.4f}'.format(auc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'auc_val: {:.4f}'.format(auc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_train.item(), acc_train.item(), ap_train.item(), auc_train.item(), loss_val.item(), acc_val.item(), ap_val.item(), auc_val.item()


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(input=output[idx_test], target=labels[idx_test], weight=list(labels[idx_test].shape)[0]/(2*torch.bincount(labels[idx_test])))
    acc_test, prec_test, recall_test, fone_test, ap_test, auc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "auc= {:.4f}".format(auc_test.item()))
    
    return acc_test.item(), prec_test.item(), recall_test.item(), fone_test.item(), ap_test.item(), auc_test.item()


cv_acc_train = []
cv_ap_train = []
cv_auc_train = []

cv_acc_val = []
cv_ap_val = []
cv_auc_val = []

cv_acc_test = []
cv_ap_test = []
cv_auc_test = []

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=123)
for tmp, idx_test in cv.split(features, labels):

    # cv = StratifiedKFold(n_splits=5)
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    for idx_train, idx_val in cv.split(features[tmp], labels[tmp]):
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        # Train model
        t_total = time.time()

        loss_tr = [None] * args.epochs
        acc_tr = [None] * args.epochs
        ap_tr = [None] * args.epochs
        auc_tr = [None] * args.epochs

        loss_val = [None] * args.epochs
        acc_val = [None] * args.epochs
        ap_val = [None] * args.epochs
        auc_val = [None] * args.epochs

        for epoch in range(args.epochs):
            loss_tr[epoch], acc_tr[epoch], ap_tr[epoch], auc_tr[epoch], loss_val[epoch], acc_val[epoch], ap_val[epoch], auc_val[epoch] = train(epoch)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        cv_acc_train = cv_acc_train + [np.max(acc_tr)]
        cv_ap_train = cv_ap_train + [np.max(ap_tr)]
        cv_auc_train = cv_auc_train + [np.max(auc_tr)]

        cv_acc_val = cv_acc_val + [np.max(acc_val)]
        cv_ap_val = cv_ap_val + [np.max(ap_val)]
        cv_auc_val = cv_auc_val + [np.max(auc_val)]

    # Testing
    acc_test, prec_test, recall_test, fone_test, ap_test, auc_test = test()
    cv_acc_test = cv_acc_test + [acc_test]
    cv_ap_test = cv_ap_test + [ap_test]
    cv_auc_test = cv_auc_test + [auc_test]

    # if(cnt == 50):
    #     plt.plot(range(args.epochs), loss_tr, label='training', linewidth=0.7)
    #     plt.plot(range(args.epochs), loss_val, label='validation', linewidth=0.7)
    #     plt.xlabel('epoch')
    #     plt.ylabel('loss')
    #     plt.legend(loc='best')
    #     plt.show()


print('cv_acc_train: {:.4f}'.format(np.mean(cv_acc_train)),
      'cv_ap_train: {:.4f}'.format(np.mean(cv_ap_train)),
      'cv_auc_train: {:.4f}'.format(np.mean(cv_auc_train)))

print('cv_acc_val: {:.4f}'.format(np.mean(cv_acc_val)),
      'cv_ap_val: {:.4f}'.format(np.mean(cv_ap_val)),
      'cv_auc_val: {:.4f}'.format(np.mean(cv_auc_val)))

print('cv_acc_test: {:.4f}'.format(np.mean(cv_acc_test)),
      'cv_ap_test: {:.4f}'.format(np.mean(cv_ap_test)),
      'cv_auc_test: {:.4f}'.format(np.mean(cv_auc_test)))

if(args.sep == 'corr_coef_pval'):
    # cut = "{}_{}_top{}".format(args.coef_cut, args.pval_cut, args.top_edges_p)
    cut = "top{}".format(args.top_edges_p)
else:
    cut = args.rand_cut


# save all training results
with open('../results/{}/{}_{}_{}_{}{}.bal.pkl'.format(args.dataset, args.dataset, args.feats, args.sep, cut, args.group), 'wb') as f:
    pickle.dump([cv_acc_train, cv_ap_train, cv_auc_train,
                 cv_acc_val, cv_ap_val, cv_auc_val,
                 cv_acc_test, cv_ap_test, cv_auc_test], f)


# plot GCN embedding (top layer)
import numpy as np
# all features (demo + mri)
# plot_embedding(features.detach().numpy(), labels, 15)

emb = np.loadtxt('../top.emb', delimiter=',')
allzeros = np.all(emb == 0, axis=1)

plot_embedding(emb=emb[~allzeros], labels=labels.numpy()[~allzeros], ppl=30, cmap_code='jet', 
            fname='../results/{}/{}_{}_{}_{}_emb{}.bal.png'.format(args.dataset, args.dataset, args.feats, args.sep, cut, args.group))

