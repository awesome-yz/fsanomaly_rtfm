"""
MIT License

Copyright (c) 2019 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F

# from crfasrnn.filters import SpatialFilter, BilateralFilter
# from crfasrnn.params import DenseCRFParams
import models
from .models import register

@register('crf')
class CrfRnn(nn.Module):
    """
    PyTorch implementation of the CRF-RNN module described in the paper:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015 (https://arxiv.org/abs/1502.03240).
    """

    def __init__(self, num_labels, num_iterations=5, crf_init_params=None, len_feats = 2048, num_seg=32, dims=512):
        """
        Create a new instance of the CRF-RNN layer.

        Args:
            num_labels:         Number of semantic labels in the dataset
            num_iterations:     Number of mean-field iterations to perform
            crf_init_params:    CRF initialization parameters
        """
        super(CrfRnn, self).__init__()

        # if crf_init_params is None:
        #     crf_init_params = DenseCRFParams()

        # self.params = crf_init_params
        self.fc = models.make(crf_init_params['fc'], **crf_init_params['fu_args'])
        self.num_iterations = num_iterations
        self._softmax = torch.nn.Softmax(dim=-1)
        self.num_labels = num_labels
        self.num_cliques = crf_init_params['num_cliques']
        self.attention_weight_layer = nn.Linear(dims, dims, bias=False, device='cuda')
        # self.compat_matrix = nn.Parameter(torch.eye(num_seg, dtype=torch.float32, device='cuda'))
        self.compat_matrix = torch.ones((num_seg, num_seg), dtype=torch.float32, device='cuda').fill_diagonal_(0)
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(len_feats),
            nn.Linear(len_feats, dims, device='cuda'))
        # self.fp = models.make(crf_init_params['fc'], **crf_init_params['fp_args'])

        # --------------------------------------------------------------------------------------------
        # --------------------------------- Trainable Parameters -------------------------------------
        # --------------------------------------------------------------------------------------------

        # Spatial kernel weights
        # self.spatial_ker_weights = nn.Parameter(
        #     crf_init_params.spatial_ker_weight
        #     * torch.eye(num_labels, dtype=torch.float32)
        # )

        # # Bilateral kernel weights
        # self.bilateral_ker_weights = nn.Parameter(
        #     crf_init_params.bilateral_ker_weight
        #     * torch.eye(num_labels, dtype=torch.float32)
        # )

        # Compatibility transform matrix
        # self.compatibility_matrix = nn.Parameter(
        #     torch.eye(num_labels, dtype=torch.float32)
        # )

    def forward(self, features):
        """
        Perform CRF inference.

        Args:
            features:  Tensor of shape (3, h, w) containing the RGB features
            logits: Tensor of shape (num_classes, h, w) containing the unary logits
        Returns:
            log-Q distributions (logits) after CRF inference
        """
        # import pdb; pdb.set_trace()
        # features = features.squeeze()
        
        batch_feat, seg_feats, len_feats = features.shape[0], features.shape[1], features.shape[-1]
        features = features.view(-1, len_feats)
        embeddings = self.embedding(features)
        embeddings = embeddings.view(batch_feat, seg_feats, -1) 
        batch_size, num_seg, len_embeddings = embeddings.shape[0], embeddings.shape[1], embeddings.shape[-1]
       
     
        skip_indx = int(embeddings.shape[1]/8)
        indexes = [i for i in range(0, 32, skip_indx)]

        pairwise_similarity = []
        for i, z in enumerate(indexes):
            x = embeddings[:,indexes[i]]
            if z == indexes[-1]:
                y = embeddings[:, indexes[0]]
                pairwise_similarity.append(F.cosine_similarity(x,y))
            else:
                y = embeddings[:,indexes[i+1]]
                # F.cosine_similarity(x,y)
                pairwise_similarity.append(F.cosine_similarity(x,y))

        # import pdb; pdb.set_trace()
        cliques =torch.stack(pairwise_similarity, dim=1)
        cliques = torch.tile(cliques, (1, num_seg)).reshape(batch_size, num_seg, self.num_cliques, 1)
        feat_embd = torch.tile(embeddings, (1,1,self.num_cliques)).reshape(batch_size, num_seg, self.num_cliques, len_embeddings)
        logits = self.fc(embeddings).squeeze()
        cur_logits = self._softmax(logits)
        # cur_logits = self._softmax(cur_logits)
        
        for _ in range(self.num_iterations):
            # Normalization
            
            # linear layer with weight vector size of the clique
            H = self.attention_weight_layer(torch.mul(feat_embd, cliques)).sum(dim=2)
            # add a softmax function
            # H = self.attention_weight_layer(h_t)
            Ep = self.fc(H).squeeze()
            Ep = torch.mm(Ep, self.compat_matrix)
            E =  cur_logits + Ep
            cur_logits = self._softmax(E)

        # import pdb; pdb.set_trace()
        return cur_logits, H
