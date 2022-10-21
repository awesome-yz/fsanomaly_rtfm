import numpy as np
import torch
import torch.nn as nn
import models
from .models import register
import utils.few_shot as fs
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import math


@register('laplacian')
class laplacian(nn.Module):

    def __init__(self, var_model_path, use_gpu=True, n_channels=3):
        super(laplacian, self).__init__()
        self.var_inference = models.load(torch.load(var_model_path))
        self.gpu = use_gpu

    def forward(self, data, n_cls, n_shot, n_query, lmd=0.5):
        
        #import pdb; pdb.set_trace()
        x_support, y_support, x_query, y_query = data[0], data[1], data[2], data[3]
        assert x_support.shape[0] == 1, 'batch size of 1 is only allowed'
        x_support, x_query = x_support.squeeze(), x_query.squeeze()
        y_support, y_query = y_support.squeeze().to(torch.float32), y_query.squeeze().to(torch.float32)
        if torch.cuda.is_available():
            x_support, x_query = x_support.cuda(), x_query.cuda()
            y_support, y_query = y_support.cuda(), y_query.cuda()
    
        # Rtfm
        #x_ab_qs, x_nor_qs = torch.cat((x_support[:n_shot], x_query[:n_query])), torch.cat((x_support[n_shot:], x_query[n_query:]))
        #x_qs = torch.cat((x_ab_qs, x_nor_qs))
        #y_ab_qs, y_nor_qs = torch.cat((y_support[:n_shot], y_query[:n_query])), torch.cat((y_support[n_shot:], y_query[n_query:]))
        #y_qs = torch.cat((y_ab_qs, y_nor_qs))
        
        score_ab, score_n, feat_abn, feat_n, scores, mean_ab, mean_n = self.var_inference.rtfm(x_support)
        #score_ab, score_n, feat_abn, feat_n, scores, mean_ab, mean_n = self.var_inference.rtfm(x_qs)
        abn_scores = scores[:n_shot]

        # VI
        #s_mean_ab, s_mean_n = mean_ab[:n_shot], mean_n[:n_shot]
        #q_mean_ab, q_mean_n = mean_ab[n_shot:], mean_n[n_shot:]
        
        s_mean_ab, s_mean_n = mean_ab, mean_n
        ab_params, n_params = self.var_inference.get_aggr_cls_params(s_mean_ab), self.var_inference.get_aggr_cls_params(s_mean_n)
        all_params = (torch.cat((ab_params[0].unsqueeze(0), n_params[0].unsqueeze(0)), dim=0),\
             torch.cat((ab_params[1].unsqueeze(0), n_params[1].unsqueeze(0)), dim=0))
        
        _,_,_,_,_, q_mean_ab, q_mean_n = self.var_inference.rtfm(x_query)
        x_query = torch.cat((q_mean_ab, q_mean_n))
        labels = y_query

        preds = self.get_laplacian_output(x_query, all_params, n_cls, lmd)
        return preds, labels

    def get_laplacian_output(self, x_query, class_params, n_cls, lmd):
        """
        x_query: x query values
        s_cls_params: class parameters (support set)
        """
        # import pdb; pdb.set_trace()
        # img_shape = x_query.shape[-3:]
        # feat = self.var_inference.encoder(x_query.view(-1, *img_shape))  #encoding features for all x_query
        #score_ab, score_n, feat_abn, feat_n, scores, mean_ab, mean_n  = self.var_inference.rtfm(x_query)
        x_query, _ = self.var_inference.g(x_query)  # generating mu for all x_query
        mu_matrix, _ = class_params  # class parameters

        m = x_query.size(0)  # number of query samples
        n = mu_matrix.size(0)  # number of class labels
        mu_matrix_dist = mu_matrix.expand(m, n, mu_matrix.size(1))
        x_query_dist = x_query.expand(n, m, x_query.size(1))
        x_query_dist = x_query_dist.transpose(0, 1)
        l2_dist = torch.pairwise_distance(mu_matrix_dist, x_query_dist)
        unary = l2_dist ** 2
        knn = n_cls
        w = self.create_affinity(x_query, knn)
        output = self.bound_update(unary, w, lmd)
        # output = np.take(output)
        output = torch.from_numpy(output)
        return output.cuda()

    def create_affinity(self, x, knn):
        N, D = x.shape
        nbrs = NearestNeighbors(n_neighbors=knn).fit(x.cpu().numpy())
        dist, knn_idx = nbrs.kneighbors(x.cpu().numpy())

        row = np.repeat(range(N), knn - 1)
        col = knn_idx[:, 1:].flatten()
        data = np.ones(x.shape[0] * (knn - 1))
        w = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=np.float)
        return w

    def normalize(self, y_in):
        maxcol = np.max(y_in, axis=1)
        y_in = y_in - maxcol[:, np.newaxis]
        N = y_in.shape[0]
        size_limit = 150000
        if N > size_limit:
            batch_size = 1280
            y_out = []
            num_batch = int(math.ceil(1.0 * N / batch_size))
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, N)
                tmp = np.exp(y_in[start:end, :])
                tmp = tmp / (np.sum(tmp, axis=1)[:, None])
                y_out.append(tmp)
            del y_in
            y_out = np.vstack(y_out)
        else:
            y_out = np.exp(y_in)
            y_out = y_out / (np.sum(y_out, axis=1)[:, None])

        return y_out

    def entropy_energy(self, Y, unary, kernel, bound_lambda, batch=False):
        tot_size = Y.shape[0]
        pairwise = kernel.dot(Y)

        if batch == False:
            temp = (unary * Y) + (-bound_lambda * pairwise * Y)
            E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()
        else:
            batch_size = 1024
            batch_size = 1024
            num_batch = int(math.ceil(1.0 * tot_size / batch_size))
            E = 0
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, tot_size)
                temp = (unary[start:end] * Y[start:end]) + (-bound_lambda * pairwise[start:end] * Y[start:end])
                E = E + (Y[start:end] * np.log(np.maximum(Y[start:end], 1e-20)) + temp).sum()

        return E

    def bound_update(self, unary, kernel, bound_lambda, bound_iteration=20, batch=False):
        oldE = float('inf')
        unary = unary.cpu().detach().numpy()  # converting to numpy
        Y = self.normalize(-unary)
        E_list = []
        for i in range(bound_iteration):
            additive = -unary
            mul_kernel = kernel.dot(Y)
            Y = - bound_lambda * mul_kernel
            additive = additive - Y
            Y = self.normalize(additive)
            E = self.entropy_energy(Y, unary, kernel, bound_lambda, batch)
            E_list.append(E)
            # print('entropy_energy is ' +repr(E) + ' at iteration ',i)
            if (i > 1 and (abs(E - oldE) <= 1e-6 * abs(oldE))):
                # print('Converged')
                break

            else:
                oldE = E.copy()
        l = np.argmax(Y, axis=1)
        return l

