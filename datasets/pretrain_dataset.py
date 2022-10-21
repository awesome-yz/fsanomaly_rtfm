from fileinput import filename
from torch.utils.data import Dataset
import numpy as np
from .datasets import register
from datasets.get_features import GetFeatures as f


@register('pretrain-dataset')
class PretrainDataset(Dataset):
    def __init__(self, abnormal_feats, normal_feats) -> None:
        super(PretrainDataset).__init__()
        self.abnormal = np.array(abnormal_feats)
        if len(self.abnormal.shape) >= 4 or len(self.abnormal.shape)==1:
            self.abnormal = np.concatenate(self.abnormal)
        self.normal = np.array(normal_feats)
        self.length = min(len(self.normal), len(self.abnormal))

    def shuffle(self):
        ab_indices = np.random.permutation(len(self.abnormal))
        no_indices = np.random.permutaion(len(self.normal))
        self.abnormal = self.abnormal[ab_indices]
        self.normal = self.normal[no_indices]

    def __getitem__(self, item):
        return np.concatenate([self.abnormal[item], self.normal[item]]), np.array([1,0])

    def __len__(self):
        return self.length
    


# if __name__ == '__main__':
#     import pdb; pdb.set_trace()

#     ab_data = f('/impacs/yuz19/data/UCF-feats', 'Abnormal')
#     no_data = f('/impacs/yuz19/data/UCF-feats', 'Normal')
    
#     ab_feats, ab_lbl = ab_data.data, ab_data.label
#     no_feats, no_lbl = no_data.data, no_data.label
    
#     data = PretrainDataset(ab_feats, no_feats)
    
#     print(data.shape)
#     print('End')                    

