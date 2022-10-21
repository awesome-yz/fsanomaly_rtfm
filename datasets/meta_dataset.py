from fileinput import filename
import torch
from torch.utils.data import Dataset
import numpy as np
from .datasets import register

@register('meta_dataset')
class MetaDataset(Dataset):
    """
    creates an episode of tasks (classes)
    contains both normal and abnormal samples
    n_cls == num_tasks : number of tasks per episodes
    n_sprt: number of support samples
    n_qry: number of query samples
    """
    def __init__(self, abnorm_feats, norm_feats, n_batch, n_cls, n_sprt=10, n_qry=30, ) -> None:
        super(MetaDataset).__init__()
        # convert list to array
        self.abnorm_feats = np.array(abnorm_feats)
        if len(self.abnorm_feats.shape) >= 4 or len(self.abnorm_feats.shape) == 1:
            self.abnormal_feats = np.concatenate(self.abnorm_feats)
        
        self.norm_feats = np.array(norm_feats)
        self.length = min(len(self.abnorm_feats), len(self.norm_feats))
        self.n_sprt = n_sprt
        self.n_qry = n_qry
        self.n_cls = n_cls
        self.n_batch = n_batch
        self.build_episode()

    def build_episode(self):
        self.ab_sprt = []
        self.no_sprt = []
        self.ab_qry = []
        self.no_qry = []
        # import pdb; pdb.set_trace()
        np.random.RandomState(seed=42)
        for i in range(self.n_batch):
            ab_indices = np.random.permutation(len(self.abnorm_feats))
            no_indices = np.random.permutation(len(self.norm_feats))

            self.ab_sprt.append(self.abnorm_feats[ab_indices[:self.n_sprt]])
            self.no_sprt.append(self.norm_feats[no_indices[:self.n_sprt]])
            self.ab_qry.append(self.abnorm_feats[ab_indices[self.n_sprt: self.n_sprt+self.n_qry]])
            self.no_qry.append(self.norm_feats[no_indices[self.n_sprt: self.n_sprt+ self.n_qry]])

    def __getitem__(self, item):
        # import pdb; pdb.set_trace()
        return torch.from_numpy(np.concatenate([self.ab_sprt[item], self.no_sprt[item]])), torch.from_numpy(np.array([1] * self.n_sprt + [0] * self.n_sprt)),\
               torch.from_numpy(np.concatenate([self.ab_qry[item], self.no_qry[item]])), torch.from_numpy(np.array([1] * self.n_qry + [0] * self.n_qry))

    def __len__(self):
        return self.n_batch 



