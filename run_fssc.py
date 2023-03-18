from time import time
import math, os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from fssc import Fssc
from single_cell_tools import *
import numpy as np
import collections
from sklearn import metrics
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize
import time

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--n_clusters', type=int, 
                        help='number of clusters')
    #parser.add_argument('--knn', default=20, type=int, 
    #                    help='number of nearest neighbors, used by the Louvain algorithm')
    #parser.add_argument('--resolution', default=.8, type=float, 
    #                    help='resolution parameter, used by the Louvain algorithm, larger value for more number of clusters')
    #parser.add_argument('--select_genes', default=0, type=int, 
    #                    help='number of selected genes, 0 means using all genes')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--sigma', default=2.5, type=float,
                        help='coefficient of random noise')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.0001, type=float,
                        help='tolerance for delta clustering labels to terminate training stage')
    parser.add_argument('--xi', default=32, type=int,
                        help='threshold for minimum number of features left')
    parser.add_argument('--omega', default=0.95, type=float,
                        help='threshold for removed features in each round')
    parser.add_argument('--epsilon', default=1.02, type=float,
                        help='penalty multiplier ')
    parser.add_argument('--ae_weights', default=None,
                        help='file to pretrained weights, None for a new pretraining')
    parser.add_argument('--save_dir', default='results/',
                        help='directory to save model weights during the training stage')
    #parser.add_argument('--ae_weight_file', default='AE_weights.pth.tar',
    #                    help='file name to save model weights after the pretraining stage')
    #parser.add_argument('--final_latent_file', default='final_latent_file.txt',
    #                    help='file name to save final latent representations')
    parser.add_argument('--predict_label_file', default='pred_labels.txt',
                        help='file name to save final clustering labels')
    parser.add_argument('--device', default='cuda')


    args = parser.parse_args()

    data_mat = h5py.File(args.data_file, 'r')
    x = np.array(data_mat['X'])
    # y is the ground truth labels for evaluating clustering performance
    # If not existing, we skip calculating the clustering performance metrics (e.g. NMI ARI)
    if 'Y' in data_mat:
        y = np.array(data_mat['Y'])
    else:
        y = None
    
    # disc_feature is the ground truth discriminatory genes
    if 'disc_feature' in data_mat:
        gt_feat = np.array(data_mat['disc_feature'])
    else:
        gt_feat = None
        
    data_mat.close()

    #if args.select_genes > 0:
    #    importantGenes = geneSelection(x, n=args.select_genes, plot=False)
    #    x = x[:, importantGenes]

    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(x)
    if y is not None:
        adata.obs['Group'] = y

    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    input_size = adata.n_vars

    print(args)

    print(adata.X.shape)
    if y is not None:
        print(y.shape)

    mask_idx = np.arange(x.shape[1])

    gene_selected_hist = []
    y_pred_hist = []
    acc_hist = []
    ari_hist = []
    nmi_hist = []
    mask_idx_removed_hist = []

    t0 = time.time()
    
    t = 0
    lambda_ = 1e-3
    while True:
        print('Round ', t)
        model = Fssc(input_dim=adata.n_vars, z_dim=32, 
                    encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=args.sigma, gamma=args.gamma, device=args.device,
                             likelihood_type='zinb')
        if (t==0):
            pre_w0_mask = model.pre_w0_mask.detach()
        else:
            model.pre_w0_mask = pre_w0_mask.detach()

        model.w_mask()
        model.pretrain_autoencoder(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                                    batch_size=args.batch_size, epochs=args.pretrain_epochs)

        z_hat= model.encodeBatch(torch.tensor(adata.X).float())

        y_pred, acc, nmi, ari = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                                                               n_clusters=args.n_clusters, init_centroid=None, 
                        y_pred_init=None, y=y, batch_size=args.batch_size, num_epochs=2000, update_interval=args.update_interval, 
                                       tol=args.tol, save_dir=args.save_dir,
                                      lambda1 = 0.)
        
        ii = 0
        lambda_ = lambda_ * 0.9
        while(True):
            print("Clustering stage ", ii)
            #print(lambda_)
            ii+=1
            mu_pre = model.mu.data.cpu().detach().numpy().copy()
            y_pred, acc, nmi, ari = model.fit(X=adata.X, X_raw=adata.raw.X,
                                           size_factor=adata.obs.size_factors, 
                                           n_clusters=args.n_clusters, init_centroid=mu_pre, 
                                           y_pred_init=y_pred.copy(), y=y, 
                                           batch_size=args.batch_size, num_epochs=100, 
                                           update_interval=args.update_interval, 
                                           tol=args.tol, save_dir=args.save_dir,
                                           lambda1 = lambda_)
            gene_selected = model.input_mask().cpu().numpy()
            print('number of selected genes ', gene_selected.sum())
            #print('\n')
            if(lambda_ <= 2e-1):
                lambda_ *= np.sqrt(10)
            else:
                lambda_ *= args.epsilon
            if(gene_selected.sum()<= max(args.xi, pre_w0_mask.sum().item()*args.omega)):
                break

        if y is not None:
            print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

        pre_w0_mask = model.input_mask().detach()

        gene_selected_hist.append(gene_selected.copy())
        y_pred_hist.append(y_pred.copy())
        acc_hist.append(acc)
        ari_hist.append(ari)
        nmi_hist.append(nmi)

        if (t==0):
            mask_idx_removed_hist.append(model.gene_removed.copy())
        else:
            removed_new = [x for x in model.gene_removed if x not in mask_idx_removed_hist[-1]]
            mask_idx_removed_hist.append(np.concatenate([mask_idx_removed_hist[-1], removed_new]))

        if(gene_selected.sum()<=args.xi):
            break

        t+=1
        print(' ')
        
    print('Total time: %d seconds.' % int(time.time() - t0))

    n_genes = [x.sum() for x in gene_selected_hist]
   
    norm_v = torch.norm(model.encoder[0].weight.data, p=2, dim=0).detach()
    idx1 = np.argsort( (1/norm_v).cpu().numpy() )[::-1]
    removed_new = [x for x in idx1 if x not in mask_idx_removed_hist[-1]]
    idx_sort = np.concatenate([mask_idx_removed_hist[-1], removed_new])
    
    df = pd.DataFrame()
    df['n_genes'] = n_genes
    df['acc'] = acc_hist
    df['ari'] = ari_hist
    df['nmi'] = nmi_hist
    df['y_pred'] = y_pred_hist
    df['gene_selected'] = gene_selected_hist
    
    df_c = df.copy()
    
    df['ari_change'] = np.nan
    for j in range(1, df.shape[0]-1):
        y_prev = df.loc[j-1, 'y_pred']
        y_cur = df.loc[j, 'y_pred']
        df.loc[j, 'ari_change'] = metrics.adjusted_rand_score(y_prev, y_cur)
        
    df['ari_change_mean'] = np.roll(df['ari_change'].rolling(5).mean().values, -2)
    idx = df['ari_change_mean'].argmax()
    
    acc, ari, y_pred_final, gene_select_final = df.loc[idx, ['acc', 'ari', 'y_pred', 'gene_selected']]

    if y is not None:
        print('Clustering results:')
        print('Accuracy: %.3f, ARI: %.3f' % (acc, ari))

    if gt_feat is not None:
        recall = metrics.recall_score(gt_feat, gene_select_final)
        prec = metrics.precision_score(gt_feat, gene_select_final)
        print('Feature selection:')
        print('Recall: %.3f, Precision: %.3f' % (recall, prec))


    h5 = h5py.File(args.predict_label_file, 'w')
    h5.create_dataset('clustering', data = y_pred_final)
    h5.create_dataset('selected_feature', data = gene_select_final)
    # from least discriminatory to most discriminatory
    h5.create_dataset('feature_index_importance_rank', data = idx_sort)
    h5.close()
    
    print(idx)
    print(df)
    