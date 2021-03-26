import os.path
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import *

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def process_data(graph_type='corr_coef_pval', group='', rand_cut=0.1, pval_cut=0.05, coef_cut=0.7, dataset='adni', feats='dma'):

    if(dataset == 'adni'):
        # 506 samples
        # <id> <74 features> <class label>
        clinical = pd.read_excel('/data1/GCN_AB/adni/CTX_Demo_n506.xlsx', engine='openpyxl')
        if(group == '.mci'):
            clinical = clinical[clinical.Dx == 1]
        elif(group == '.nc'):
            clinical = clinical[clinical.Dx == 0]

        id = clinical.loc[:, "ID"]
        demo = clinical.loc[:, ["Age","Sex", "Edu", "APOEe4"]]
        mri = clinical.loc[:, "SupraTentorial":"RightMiddleTemporal"]

        label = clinical.loc[:, "111CUTOFF"]

        if(feats == 'dm'):
            feat = feat.drop(['APOEe4'], axis=1)

        # prs = pd.read_csv('{}prs_output_IGAP.profile'.format(path), delimiter=r'\s+')

    elif(dataset == 'smc'):
        # 521 samples
        # <id> <25 features> <class label>
        clinical = pd.read_excel('/data1/GCN_AB/smc/CTX_Demo_n521.xlsx', engine='openpyxl')
        if(group == '.mci'):
            clinical = clinical[clinical.ABPET_Dx == 1]
        elif(group == '.nc'):
            clinical = clinical[clinical.ABPET_Dx == 0]

        id = clinical.loc[:, "Hos"]
        demo = clinical.loc[:, ["test_age", "sex", "education", "APOE4"]]
        mri = clinical.loc[:, "ICV.mm3.":"HV_native_R"]

        label = clinical.loc[:, "amyloid_beta"]

        if(feats == 'dm'):
            feat = feat.drop(['APOE4'], axis=1)

    
    feat = pd.concat([demo, mri], axis=1)     
    feat = (feat - feat.min()) / (feat.max() - feat.min())
    print(feat.shape)

    df = pd.concat([id, feat, label], axis=1)
    df.to_csv("/data1/GCN_AB/{}/{}_{}{}.content".format(dataset, dataset, feats, group), sep=' ', index=False, header=False)

    if(graph_type == 'random'):
        edge_cut = int(round(rand_cut/100*(len(id)*len(id)-len(id))))

        o = np.ones(edge_cut, dtype=np.int)
        z = np.zeros(np.product((len(id), len(id))) - edge_cut, dtype=np.int)
        board = np.concatenate([o, z])
        np.random.shuffle(board)
        board = board.reshape((len(id), len(id)))

        graph = nx.from_pandas_adjacency(pd.DataFrame(board, index=id, columns=id))

        # graph = nx.binomial_graph(n=506, p=edge_cut/((506*506)/2), seed=22)
        # graph = nx.relabel_nodes(graph, {new:old for new,old in enumerate(df.ID)}, copy=False)

        graph.remove_edges_from(nx.selfloop_edges(graph))
        print(nx.info(graph))
        #plot_degree_dist(graph)

        nx.write_edgelist(graph, "/data1/GCN_AB/{}/{}_{}_{}_{}{}.edges".format(dataset, dataset, feats, graph_type, rand_cut, group), data=False)

    elif(graph_type == 'corr_coef_pval'):
        from scipy.stats import pearsonr
        corr_df = feat.T
        corr_df.columns = id

        corr_coef_df = corr_df.corr()
        corr_pval_df = corr_df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(len(corr_df.columns))
        
        df1 = corr_coef_df.rename_axis('', axis=0).stack().reset_index()
        df1.columns = ['s', 't', 'w']

        df2 = corr_pval_df.rename_axis('', axis=0).stack().reset_index()
        df2.columns = ['s', 't', 'p']

        merged_df = pd.merge(df1, df2, on=['s', 't'])
        print(merged_df.head())

        merged_df.drop(merged_df[merged_df.s == merged_df.t].index, inplace=True)
        print('df1 {}, df2 {}, merged {}'.format(df1.shape[0], df2.shape[0], merged_df.shape[0]))

        # graph = nx.from_pandas_adjacency((np.abs(corr_coef_df) > 0.9).astype(int))
        # graph = nx.from_pandas_adjacency((corr_pval_df < edge_cut).astype(int))
        # graph = nx.from_pandas_adjacency(((np.abs(corr_coef_df) > coef_cut) & (corr_pval_df < pval_cut)).astype(int))
        # graph = nx.from_pandas_adjacency(corr_coef_df[corr_pval_df < pval_cut].fillna(0))

        # graph.remove_edges_from(nx.selfloop_edges(graph))
        # print(nx.info(graph))
        # print('sparsity {:.4f}%'.format(graph.number_of_edges() / (graph.number_of_nodes() * graph.number_of_nodes()) * 100))

        # plot_degree_dist(graph)

        # edge_cut = "{:.1f}_{}".format(coef_cut, pval_cut)
        edge_cut = 'weighted'
        fname = "/data1/GCN_AB/{}/{}_{}_{}_{}{}.edges".format(dataset, dataset, feats, graph_type, edge_cut, group)
        
        # nx.write_edgelist(graph, fname, data=['weight'])
        merged_df.to_csv(fname, sep=' ', header=False, index=False)
        sort_edgelist(fname)

    # nx.draw_networkx(graph_pval, with_labels=False, node_size=30)


def sort_edgelist(fname):
    edgelist = pd.read_table(fname, header=None, sep=' ')
    edgelist.columns = ['s', 't', 'w', 'p']
    edgelist = edgelist.sort_values(by=['w', 'p'], ascending=[False, True])
    edgelist.to_csv(fname, sep=' ', header=False, index=False)

def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.show()


def load_data(dataset="adni", feats="dma", graph="corr_coef_pval", group="", rand_cut=0.1, pval_cut=0.05, coef_cut=0.7, top_edges_p=100):
    
    print('Loading {} dataset...'.format(dataset))

    if(graph == 'corr_coef_pval'):
        # cut = "{:.1f}_{}".format(coef_cut, pval_cut)
        cut = 'weighted'
    else:
        cut = rand_cut
        
    content_file = "/data1/GCN_AB/{}/{}_{}{}.content".format(dataset, dataset, feats, group)
    edge_file = "/data1/GCN_AB/{}/{}_{}_{}_{}{}.edges".format(dataset, dataset, feats, graph, cut, group)

    if((os.path.isfile(content_file) == False) | (os.path.isfile(edge_file) == False)):
        print('Processing {} dataset...'.format(dataset))
        process_data(graph_type=graph, group=group, rand_cut=rand_cut, pval_cut=pval_cut, coef_cut=coef_cut, dataset=dataset, feats=feats)
    
    idx_features_labels = np.genfromtxt(content_file, dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    # build graph
    adj=""
    
    idx_map = {j: i for i, j in enumerate(idx)}
    
    edges_unordered = np.genfromtxt(edge_file, dtype=np.int32, usecols=[0,1])

    top_edges = int(round(np.shape(edges_unordered)[0] * top_edges_p / 100))
    edges_unordered = edges_unordered[:top_edges, :]
    # weights = np.genfromtxt(edge_file, dtype=np.float64, usecols=[2], max_rows=top_edges)
    
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # adj = sp.coo_matrix((weights, (edges[:,0], edges[:,1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float64)

    print(len(adj.data))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = normalize(adj + sp.eye(adj.shape[0]))

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, idx



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    accuracy = correct / len(labels)

    preds = output.max(1)[1]


    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    fone = f1_score(labels, preds)
    ap = average_precision_score(labels, preds)
    auc = roc_auc_score(labels, preds)

    return accuracy, precision, recall, fone, ap, auc



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def plot_embedding(emb, labels, ppl, fname, cmap_code='jet'):
    from sklearn.manifold import TSNE

    trans = TSNE(n_components=2, perplexity=ppl)
    emb = trans.fit_transform(emb)

    import matplotlib.pyplot as plt
    plt.scatter(
        emb[:, 0],
        emb[:, 1],
        c=labels,
        cmap=cmap_code,
        alpha=0.7,
    )
    plt.savefig(fname)

def classification_test(X, y, CFs, show_plot=False, k=10, rep=10, dict={}):

    if(show_plot):
        f, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    for CF in CFs:
        aucs = []

        cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=rep)

        y_real = []
        y_pred = []
        y_prob = []
        for train_id, test_id in cv.split(X, y):
            X_train, X_test = X[train_id], X[test_id]
            y_train, y_test = y[train_id], y[test_id]

            if CF == "SVC":
                model = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42)
                
            elif CF == "RFC":
                model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
                
            elif CF == "LR-lasso":
                model = LogisticRegression(penalty="l1", solver="liblinear", class_weight='balanced', random_state=42)
                
            elif CF == "LR-ridge":
                model = LogisticRegression(penalty="l2", solver="liblinear", class_weight='balanced', random_state=42)
                
            elif CF == "MLP":
                model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=550, learning_rate_init=0.005, alpha=5e-10,
                                      early_stopping=False, validation_fraction=0.2, class_weight='balanced', random_state=42)

            clf = model.fit(X_train, y_train)
            pred = clf.predict(X_test)
            prob = clf.predict_proba(X_test)[:, 1]

            y_real.append(y_test)
            y_pred.append(pred)
            y_prob.append(prob)

            fpr, tpr, _ = roc_curve(y_test, prob)
            aucs.append(auc(fpr, tpr))

        dict[CF] = aucs

        y_real = np.concatenate(y_real)
        y_pred = np.concatenate(y_pred)
        y_prob = np.concatenate(y_prob)

        if (show_plot):
            fpr, tpr, _ = roc_curve(y_real, y_prob)
            lab = '%s (%.4f)' % (CF, auc(fpr, tpr))
            axes[0].step(fpr, tpr, label=lab, lw=1)

            precision, recall, _ = precision_recall_curve(y_real, y_prob)
            lab = '%s (%.4f)' % (CF, auc(recall, precision))
            axes[1].step(recall, precision, label=lab, lw=1)

            axes[0].set_xlabel('FPR')
            axes[0].set_ylabel('TPR')
            axes[0].set_ylim(0, 1.05)
            axes[0].legend(loc='lower right', fontsize='small')

            axes[1].set_xlabel('Recall')
            axes[1].set_ylabel('Precision')
            axes[1].set_ylim(0, 1.05)
            axes[1].legend(loc='lower right', fontsize='small')


    return dict
