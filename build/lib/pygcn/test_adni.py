from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()

# 506 samples
# 506 x 506
dist = pd.read_excel('data/DMx.xlsx', index_col=0, header=None)
dist.columns = dist.index
sigma = np.mean(dist)
dist = np.exp(- dist ** 2 / (2 * sigma ** 2))

# graph = nx.from_pandas_adjacency((dist > 0.7).astype(int))
graph = nx.from_pandas_adjacency(dist)
print(nx.info(graph))
adj = nx.adjacency_matrix(graph)

deg = dict(nx.degree(graph))
h = plt.hist(deg.values(), 100)

plt.loglog(h[1][1:], h[0])

# 506 x 83
clinical = pd.read_excel('data/CTX_Demo_n506.xlsx')
demo = clinical.loc[:, ["Age", "Sex", "Edu", "APOEe4"]]
ctx = clinical.loc[:, "SupraTentorial":"RightMiddleTemporal"]
label = clinical.loc[:,"111CUTOFF"]

# 30 labeled in training (5% of all)
# 306 labeled/unlabeled training
# 100 validation
# 200 test
n_tr = 356
n_val = 50
n_te = 100

feat = pd.concat([demo, ctx], axis=1)
feat.index = clinical.FAM2
feat = (feat-feat.mean())/feat.std()

features = sp.csr_matrix(feat).tolil()

labels = np.column_stack([np.asarray([label == 0], dtype=np.int).T,
                          np.asarray([label == 1], dtype=np.int).T])

rand = np.random.permutation(len(labels))
idx_train = rand[range(n_tr)]
idx_val = rand[range(n_tr, n_tr + n_val)]
idx_test = rand[range(len(labels)-n_te, len(labels))]

# idx_train = range(n_tr)
# idx_val = range(n_tr, n_tr + n_val)
# idx_test = range(len(labels)-n_te, len(labels))

train_mask = sample_mask(idx_train, labels.shape[0])
val_mask = sample_mask(idx_val, labels.shape[0])
test_mask = sample_mask(idx_test, labels.shape[0])

y_train = np.zeros(labels.shape)
y_val = np.zeros(labels.shape)
y_test = np.zeros(labels.shape)
y_train[train_mask, :] = labels[train_mask, :]
y_val[val_mask, :] = labels[val_mask, :]
y_test[test_mask, :] = labels[test_mask, :]

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
