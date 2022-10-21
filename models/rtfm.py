import re
import torch
import torch.nn as nn
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.FloatTensor')
import models
from .models import register


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]


        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        # import pdb; pdb.set_trace()    
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z

class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1,
                      stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(512, sub_sample=False, bn_layer=True)

    def forward(self, x):
            # x: (B, T, F)
            # import pdb; pdb.set_trace()
            out = x.permute(0, 2, 1)
            residual = out

            out1 = self.conv_1(out)
            out2 = self.conv_2(out)

            out3 = self.conv_3(out)
            out_d = torch.cat((out1, out2, out3), dim = 1)
            out = self.conv_4(out)
            out = self.non_local(out)
            out = torch.cat((out_d, out), dim=1)
            out = self.conv_5(out)   # fuse all the features together
            out = out + residual
            out = out.permute(0, 2, 1)
            # out: (B, T, 1)

            return out
@register('rtfm')
class Model(nn.Module):
    def __init__(self, rtfm_sd, n_features, batch_size):
        super(Model, self).__init__()
        #import pdb; pdb.set_trace()
        self.batch_size = batch_size
        # self.abn_split = batch_size /2
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        self.Aggregate = Aggregate(len_feature=2048)
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        #self.apply(weight_init)
        self.load_state_dict(torch.load(rtfm_sd))

    def forward(self, inputs):

        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs
        #bs, ncrops, t, f = out.size()
        bs, t, f = out.size()
        abn_split = bs // 2

        # out = out.view(-1, t, f)

        out = self.Aggregate(out)

        out = self.drop_out(out)

        features = out
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        # # scores = scores.view(bs, ncrops, -1).mean(1)  # need to check this
        # scores = scores.view(bs, -1).mean(1)
        # scores = scores.unsqueeze(dim=1)

        # normal_features = features[0:self.batch_size*10]
        # normal_scores = scores[0:self.batch_size]

        # abnormal_features = features[self.batch_size*10:]
        # abnormal_scores = scores[self.batch_size:]

        abnormal_features = features[0:abn_split]
        abnormal_scores = scores[0:abn_split]

        normal_features = features[abn_split:]
        normal_scores = scores[abn_split:]

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        # feat_magnitudes = feat_magnitudes.view(bs, -1).mean(1)
        # nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        # afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes
        afea_magnitudes = feat_magnitudes[0: abn_split]
        nfea_magnitudes = feat_magnitudes[abn_split:]
        n_size = nfea_magnitudes.shape[0]

        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        #######  process abnormal videos -> select top3 feature magnitude  #######
        select_idx = torch.ones_like(afea_magnitudes)
        select_idx = self.drop_out(select_idx)
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]

        abnormal_features = abnormal_features.view(n_size, t, f)

        total_select_abn_feature = torch.zeros(0, device=inputs.device)
        total_select_abn_score = torch.zeros(0, device=inputs.device)


        # import pdb; pdb.set_trace()
        for idx, abnormal_feature in enumerate(abnormal_features):
            feat_select_abn = abnormal_feature[idx_abn[idx]]
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        # import pdb; pdb.set_trace()
        for idx, abn_score in enumerate(abnormal_scores):
            score_select_abn = abn_score[idx_abn[idx]]
            total_select_abn_score = torch.cat((total_select_abn_score, score_select_abn), dim=1)

        score_abnormal = total_select_abn_score.transpose(1,0).mean(dim=-1)


        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes)
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        # import pdb; pdb.set_trace()
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
     

        normal_features = normal_features.view(n_size, t, f)
  
        total_select_nor_feature = torch.zeros(0, device=inputs.device)
        total_select_nor_score = torch.zeros(0, device=inputs.device)

        for idx, nor_feat in enumerate(normal_features):
            feat_select_normal = nor_feat[idx_normal[idx]]  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))


        # import pdb; pdb.set_trace()
        for idx, nor_score in enumerate(normal_scores):
            score_select_normal = nor_score[idx_normal[idx]]
            total_select_nor_score = torch.cat((total_select_nor_score, score_select_normal), dim=-1)

        # import pdb; pdb.set_trace()
        score_normal = total_select_nor_score.transpose(1,0).mean(dim=-1)

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        feat_mean_abn = feat_select_abn.view(-1, self.k_abn, feat_select_abn.size(-1)).mean(dim=1)
        feat_mean_normal = feat_select_normal.view(-1, self.k_nor, feat_select_normal.size(-1)).mean(dim=1)

        # score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes
        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores, feat_mean_abn, feat_mean_normal