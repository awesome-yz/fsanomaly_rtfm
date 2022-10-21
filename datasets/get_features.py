import os
import numpy as np
import torch
import utils
from torch.utils.data import Dataset

class GetFeatures(Dataset):
   
    def __init__(self, root_path, filename, transform=None, ten_crop=False, **kwargs) -> None:
        super(GetFeatures).__init__()
        self.data = []
        self.transform = transform
        self.ten_crop = ten_crop

        # import pdb; pdb.set_trace()
        files = sorted(os.listdir(os.path.join(root_path, filename)))
        
        for i, f in enumerate(files):
            # self.data.append(np.load(os.path.join(root_path, filename, f)))
            features = np.load(os.path.join(root_path, filename, f))
            features = np.array(features, dtype=np.float32)

            if self.transform is not None:
                features = self.transform(features)
            # if self.test_mode:
            #   return features

            elif self.ten_crop:
                features = features.transpose(1, 0, 2) # [10, B, T, F] T = segments, F = features
                divided_features = []
                for feature in features:
                    feature = utils.process_feat(feature, 32)
                    divided_features.append(feature)
                features = np.array(divided_features, dtype=np.float32)
            else:
                features = utils.process_feat(features, 32)
                features = np.array(features, dtype=np.float32)
            
            # import pdb; pdb.set_trace()
            self.data.append(features)

        if filename.startswith('Ab'):
            self.label = [1] * len(files)
        else:
            self.label = [0] * len(files)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        data = self.data[i]
        label = self.label[i]
        return data, label



