import numpy as np
import torch
import torch.nn as nn
import models
from .models import register
import utils.few_shot as fs
from torch.nn import functional as F


@register('var_inference')
class VI(nn.Module):

    # deleted 'gen_args' parameters
    def __init__(self, n_way, n_shot, n_query, rtfm_args, g_args={'g_input': 2048}, use_gpu=True, n_channels=3):
        super(VI, self).__init__()
        # self.encoder = models.make(encoder, **encoder_args)
        # self.fc = models.make(fc, **fc_args)
        # self.crf = models.make(crf, **crf_args)
        #import pdb; pdb.set_trace()
        self.rtfm = models.make('rtfm', **rtfm_args)
        self.g = models.make('generator', **g_args)
        self.gpu = use_gpu
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

    def forward(self, data):
        """
        datax = images (28*28)
        datay = labels
        Ns = number of support samples in each class (k-shot)
        Nc = number of classes per episode (C-view)
        Nq = number of query samples in each class
        total_classes = number of unique labels in the train dataset
        """
        # import pdb; pdb.set_trace()
        x_support, y_support, x_query, y_query = data[0], data[1], data[2], data[3]
        assert x_support.shape[0] == 1, 'batch size of 1 is only allowed'
        x_support, x_query = x_support.squeeze(), x_query.squeeze()
        y_support, y_query = y_support.squeeze().to(torch.float32), y_query.squeeze().to(torch.float32)
        if torch.cuda.is_available():
            x_support, x_query = x_support.cuda(), x_query.cuda()
            y_support, y_query = y_support.cuda(), y_query.cuda()

        # RTFM network
        # x_ab_qs, x_nor_qs = torch.cat((x_support[:self.n_shot], x_query[:self.n_query])), torch.cat((x_support[self.n_shot:], x_query[self.n_query:]))
        # x_qs = torch.cat((x_ab_qs, x_nor_qs))

        # y_ab_qs, y_nor_qs = torch.cat((y_support[:self.n_shot], y_query[:self.n_query])), torch.cat((y_support[self.n_shot:], y_query[self.n_query:]))
        # y_qs = torch.cat((y_ab_qs, y_nor_qs))

        # score_ab, score_n, feat_abn, feat_n, scores, mean_ab, mean_n = self.rtfm(x_qs)
        score_ab, score_n, feat_abn, feat_n, scores, mean_ab, mean_n = self.rtfm(x_support)
        # abn_scores = scores[:self.n_shot + self.n_query]
        abn_scores = scores[:self.n_shot]
        # import pdb; pdb.set_trace()
        # s_mean_ab, s_mean_n = mean_ab[:self.n_shot], mean_n[:self.n_shot]
        # q_mean_ab, q_mean_n = mean_ab[self.n_shot:], mean_n[self.n_shot:]
        s_mean_ab, s_mean_n = mean_ab, mean_n

        # loss_rtfm = self.rtfm_loss(score_ab, score_n, y_qs, mean_ab, mean_n)
        # loss_rtfm = self.rtfm_loss(score_ab, score_n, y_support, mean_ab, mean_n)
        # loss_sparse = self.sparsity(abn_scores, 8e-3)
        # loss_smooth = self.smooth(abn_scores, 8e-4)
        # total_rtfm_loss = loss_rtfm + loss_sparse + loss_smooth

        # variational model
        ab_params, n_params = self.get_aggr_cls_params(s_mean_ab), self.get_aggr_cls_params(s_mean_n)
        all_params = (torch.cat((ab_params[0].unsqueeze(0), n_params[0].unsqueeze(0)), dim=0),\
             torch.cat((ab_params[1].unsqueeze(0), n_params[1].unsqueeze(0)), dim=0))
        
        true_labels = y_query

        _,_,_,_,_, q_mean_ab, q_mean_n = self.rtfm(x_query)
        x_query = torch.cat((q_mean_ab, q_mean_n))
        logits = self.get_logits(x_query, all_params)
        pred_lbl = self.pred_prob(logits, true_labels)

        # import pdb; pdb.set_trace()
        x_support = torch.cat((s_mean_ab, s_mean_n))
        intra_loss = 0.00025 * self.intra_loss(x_query, pred_lbl, x_support, y_support, all_params)
        rec_loss = (0.5 * (pred_lbl - true_labels)**2).mean()
        # loss = intra_loss + rec_loss + total_rtfm_loss
        loss = intra_loss + rec_loss

        return pred_lbl, true_labels, loss

    def rtfm_loss(self, score_abn, score_nor, label, feat_abn, feat_nor, alpha=0.0001, margin=100,):
        # import pdb; pdb.set_trace()
        bce_loss = torch.nn.BCELoss()
        score = torch.cat((score_abn, score_nor), dim=0)
        label = label
        loss_cls = bce_loss(score, label)
        # loss_abn = torch.abs(margin - torch.norm(torch.mean(feat_abn, dim=1), p=2, dim=1))
        loss_abn = torch.abs(margin - torch.norm(feat_abn, p=2, dim=0))
        # loss_nor = torch.norm(torch.mean(feat_nor, dim=1), p=2, dim=1)
        loss_nor = torch.norm(feat_nor, p=2, dim=0)
        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)
        loss_total = loss_cls + alpha * loss_rtfm

        return loss_total

    def sparsity(self, arr, lamda2):
        loss = torch.mean(torch.norm(arr, dim=0))
        return lamda2 * loss

    def smooth(self, arr, lamda1):
        arr2 = torch.zeros_like(arr)
        arr2[:-1] = arr[1:]
        arr2[-1] = arr[-1]

        loss = torch.sum((arr2-arr) ** 2)
        return lamda1 * loss

    def cal_contrastive_loss(self, support, query):
        # import pdb; pdb.set_trace()
        len_support = support.shape[0]
        crd = support.shape[1]
        s_part = int(len_support/2)
        len_query = query.shape[0]
        q_part = int(len_query/2)
        pos_support, neg_support = support[:s_part], support[s_part:]
        pos_support = pos_support.repeat(s_part,1 ,1)
        neg_support = torch.repeat_interleave(neg_support, repeats=s_part, dim=0)
        pos_query, neg_query = query[q_part:], query[:q_part]
        pos_query = pos_query.repeat(q_part, 1, 1)
        neg_query = torch.repeat_interleave(neg_query, repeats=q_part, dim=0)
        support_loss = self.contrastive_loss(pos_support, neg_support, crd, s_part)
        query_loss = self.contrastive_loss(pos_query, neg_query, crd, q_part)
        t_loss = support_loss + query_loss
        t_loss = t_loss.mean()
        return t_loss
    
    def contrastive_loss(self, pos, neg, cardinality, n_pairs):
        # import pdb; pdb.set_trace()
        norm = (cardinality * cardinality) ** 2
        pair_norm = n_pairs ** 2
        loss = torch.mul((1/norm), torch.cdist(pos, neg).sum(-1).sum(-1))
        loss = loss.sum(0)/pair_norm
        return loss

    
    def ts_loss(self, support, query, alpha=0.00008):
        # import pdb; pdb.set_trace()
        len_support = support.shape[0]
        s_part = int(len_support / 2)
        len_query = query.shape[0]
        q_part = int(len_query / 2)

        ab_support = support[:s_part]
        ab_query = query[:q_part]
        abnorm = torch.cat([ab_support, ab_query])

        # sp = torch.sum(abnorm, dim=-1)
        # sp = alpha * torch.sum(sp)

        z1 = torch.ones_like(abnorm)
        z2 = torch.cat([z1, abnorm], dim=1)
        z3 = torch.cat([abnorm, z1], dim=1)
        z_22 = z2[:, 31:]
        z_44 = z3[:, :33]
        z = z_22 - z_44
        z = z[:, 1:32]
        z = torch.sum(torch.square(z), dim=-1)
        ts = alpha * torch.sum(z)

        return ts

    def get_aggr_cls_params(self, x):
        # import pdb; pdb.set_trace()
        # n = s_cls.shape[-4]
        # shot_shape = x.shape[:-3]
        # img_shape = x.shape[-3:]
        feat_size = x.shape[-1]
        shot_shape = x.shape[:-1]
        # seg_size = x.shape[-2]
        # batch_size = x.shape[-]

        # saving all class means
        # feat = self.encoder(s_cls.view(-1, *img_shape))
        mu, log_var = self.g(x.view(-1, feat_size))

        mu = mu.view(*shot_shape, -1)
        log_var = log_var.view(*shot_shape, -1)

        # import pdb; pdb.set_trace()

        agg_mu = (mu/log_var).sum((0))/(1/log_var).sum((0))
        agg_log_var = self.n_way/(1/log_var).sum((0))
        # agg_log_var = self.n_way / torch.linalg.inv(log_var).sum((0))

        return agg_mu, agg_log_var

    def get_logits(self, x_query, class_params):
        """
        x_query: x query values
        s_cls_params: class parameters (support set)
        """
        # import pdb; pdb.set_trace()
        feat_shape = x_query.shape[-1]
        shot_shape = x_query.shape[:-1]
        x_query = x_query.view(-1, feat_shape)
        # feat = self.encoder(x_query_img)  # encoding features for all x_query
        mu_query, _ = self.g(x_query)  # generating mu for all x_query
        mu_matrix, logvar_matrix = class_params  # class parameters
        mu_matrix = mu_matrix.view(-1, mu_matrix.shape[-1])
        logvar_matrix = logvar_matrix.view(-1, logvar_matrix.shape[-1])
        m = x_query.size(0)  # number of query samples
        n = mu_matrix.size(0)  # number of class labels

        log_pxz_matrix = torch.zeros((m, n)).cuda()
        # if self.gpu:
        #     log_pxz_matrix = log_pxz_matrix.cuda()
        for j in range(n):
            log_pxz_matrix[:, j] = self.gaussian_likelihood(mu_matrix[j], logvar_matrix[j], mu_query)
        # import pdb; pdb.set_trace()
        # logits = log_pxz_matrix.view(*shot_shape, -1)
        return log_pxz_matrix.view(*shot_shape, -1)

    def pred_prob(self, logits, true_labels):
        # import pdb; pdb.set_trace()
        pred = torch.softmax(logits, dim=-1)
        pred_val = torch.argmax(pred, dim=-1)
        return pred_val
    
    def wassertein_loss(self, q, p, samples=1):
        q_mu, q_logvar = q  # class_dist based on support
        p_mu, p_logvar = p  # class_dist based on support and query

        zq = self.z(q_mu, q_logvar)
        zp = self.z(p_mu, p_logvar)

        wl = -zp * zq
        return torch.sum(wl, dim=-1)


    def z(self, mu, log_var):
        """
        Monte carlo sample
        """
        # sigma = torch.log(1 + torch.exp(log_var))
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        z_samples = mu + sigma * eps
        return z_samples

    def gaussian_likelihood(self, mu, logvar, x):
        # import pdb; pdb.set_trace()

        sigma = torch.exp(0.5 * logvar) + 1e-5
        mean = mu
        d = x.shape[-1]
        # dist = torch.distributions.Normal(mean, sigma)
        # log_pxz = dist.log_prob(x)
        # log_pxz = log_pxz.sum(-1)

        x = -0.5 * ((x - mu) ** 2 / logvar).sum(-1)
        y = -1 * torch.log(sigma).sum(-1)
        z = -0.5 * d * np.log(2 * np.pi)

        log_pxz = x + y + z
        return log_pxz

    def intra_loss(self, x_query, query_lbl, x_support, support_lbl, s_cls_params):
        """
        Calculates intra kl divergence for all classes between posterior (support set)
        and prior distrbutions (support + query set)
        Query_x: query instances
        pred: softmax predictions of query instances
        Support_x: support instances
        s_cls_params: support set aggregated class distributions parameters
        cls_labels: true labels
        Ns: number of support samples in each class
        """
        # import pdb; pdb.set_trace()
        # cls_mu = []
        cls_mu = torch.empty(0, device='cuda')
        cls_logvar = torch.empty(0, device='cuda')
        feat_shape = x_support.shape[-1]

        for cls in range(0, self.n_way):
            # getting correct labels for anomalous and normal instances
           
            idx = 1-cls # taking anomalous samples first
            s_x = x_support[support_lbl == idx]
            q_x = x_query[query_lbl == idx]
            sq_x = torch.cat([q_x.view(-1, feat_shape), s_x.view(-1, feat_shape)], dim=0)
            sq_mu, sq_log_var = self.get_aggr_cls_params(sq_x)
            cls_mu = torch.cat((cls_mu, sq_mu.view(1, -1)))
            cls_logvar = torch.cat((cls_logvar, sq_log_var.view(1, -1)))
            # sq_mu, sq_log_var = sq_mu.mean(0), sq_log_var.mean(0)
            # s_mu, s_log_var = s_cls_params[0][idx], s_cls_params[1][idx]

            # # cls_kl += self.kl_divergence((s_mu, s_log_var), (sq_mu, sq_log_var))
            # kl_div = self.kl_divergence((sq_mu, sq_log_var))
            # cls_kl.append(kl_div)
            #
        # cls_kl = torch.stack(cls_kl, dim=0)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + cls_logvar - cls_mu ** 2 - cls_logvar.exp(), dim=1), dim =0)
        # return torch.mean(cls_kl)
        return kld_loss

    def kl_divergence(self, p):
        # import pdb; pdb.set_trace()
        p_mu, p_logvar = p  # class_dist based on support and query

        p = torch.distributions.Normal(p_mu, torch.exp(0.5 * p_logvar))
        q = torch.distributions.Normal(torch.zeros_like(p_mu), torch.ones_like(p_logvar))
        z = p.rsample()

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qz - log_pz)
        kl = kl.sum(-1) # large values

        return kl




    