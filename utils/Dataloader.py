import torch
import os
from torch.utils.data import Dataset
import scipy.io as scio
import numpy as np

import math


class Dataset(Dataset):
    def __init__(self, root_path):
        super(Dataset, self).__init__()
        self.root_path = root_path
        files = os.listdir(root_path)
        self.components = []
        for file in files:
            path = self.root_path + file
            # info = np.load(path, allow_pickle=True)
            # data = info.tolist()

            self.components.append({
                'path': path,
            })

    def __getitem__(self, index):
        components = self.components[index]
        info = np.load(components['path'], allow_pickle=True)
        data = info.tolist()
        # For Self Generating data
        patch = data[0]
        label32 = data[1]
        coord32 = data[2]


        patch = torch.from_numpy(patch).to(torch.float)
        label32 = torch.from_numpy(np.array(label32)).to(torch.float)
        coord32 = torch.from_numpy(np.array(coord32)).to(torch.float)

        # patch = torch.unsqueeze(patch, 0)
        # label128 = torch.unsqueeze(label128, 0)
        # label64 = torch.unsqueeze(label64, 0)
        # label32 = torch.unsqueeze(label32, 0)
        # label16 = torch.unsqueeze(label16, 0)
        # coord128 = torch.unsqueeze(coord128, 0)
        # coord64 = torch.unsqueeze(coord64, 0)
        # coord32 = torch.unsqueeze(coord32, 0)
        # coord16 = torch.unsqueeze(coord16, 0)

        return patch, label32, coord32

    def __len__(self):
        return len(self.components)

