import numpy as np
import os
import torch

from feature_extract import extract_feature_from_pose

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, train_dir, classes):
        self.num_classes = len(classes)
        self.poses = []
        for class_idx, class_name in enumerate(classes):
            num_poses_per_class = 0
            filenames = os.listdir(os.path.join(train_dir, class_name))
            for filename in filenames:
                full_filename = os.path.abspath(os.path.join(train_dir, class_name, filename))
                if os.path.isfile(full_filename):
                    self.poses.append((full_filename, class_idx))
                    num_poses_per_class += 1
            print(f'{num_poses_per_class} poses of class {class_name}')
        print(f'{len(self.poses)} poses in total')
  
    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        filename, class_idx = self.poses[idx]
        file = open(filename, 'rb')
        pose = np.load(file)
        feat = extract_feature_from_pose(pose)
        one_hot = np.zeros((self.num_classes,), dtype=np.float32)
        one_hot[class_idx] = 1
        return torch.Tensor(feat), torch.Tensor(one_hot)
