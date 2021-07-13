# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
from dataset_loader import kitti_loader
from torch.utils.data import DataLoader
from module.network import U_Net
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

pcDir="/media/furqan/Data/Projects/PointCloud/Dataset"
serverDir="/root/dataset/kitti"



# dataset = kitti_loader(data_dir=cfg.root_dir, point_cloud_files=cfg.point_cloud_files,
#                        data_type=args.dataset_type, labels_files=cfg.labels_files,
#                        train=True, skip_frames=1)
# dataloader = DataLoader(dataset, batch_size=cfg.batch_size * cfg.num_gpus, shuffle=True,
#                         num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
# test_dataset = kitti_loader(data_dir=cfg.root_dir, point_cloud_files=cfg.point_cloud_files,
#                             data_type=args.dataset_type, labels_files=cfg.labels_files,
#                             train=False)
# test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size * cfg.num_gpus, shuffle=False,
#                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
# dt = np.dtype(['x','y','z','i'])
def plot2d(twod_array,title,cmap):
    plt.figure(figsize=(15, 2))
    plt.title(title)
    plt.imshow(twod_array, cmap=cmap)
    plt.show()

dataset = kitti_loader()
# scan,_ =dataset.__getitem__(index=1)
print("The length of dataset is = ", len(dataset))
# proj_remission, proj_sem_label, proj_xyz, proj_range, proj_idx, proj_mask = dataset[2250]
# plot2d(proj_remission, "proj_remission", "tab20c")
# plot2d(proj_sem_label,"proj_sem_label", "tab20c")
#
# # Masking cars only in 2D
# proj_label = np.zeros((64, 1024), dtype=np.float32)
# proj_label[proj_sem_label == 10] = 1
# print(proj_label[18, 0:200])
#
# plot2d(proj_label,"proj_label","Greys")


dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
#
model= U_Net()
print("Model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
#
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.cuda()
# loss_crs = ClsLoss(ignore_index=0, reduction='mean')
loss_crs = nn.CrossEntropyLoss(ignore_index=0, reduction='mean').cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.35, patience=5, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                       eps=1e-08)
def train():
    for batch_idx, (proj_remission, proj_sem_label) in enumerate(dataloader):

        print(proj_remission.shape)
        print("Batch == ", batch_idx)
        optimizer.zero_grad()
        out = model(proj_remission)
        loss = loss_crs(out, proj_sem_label)
        loss.backward()
        optimizer.step()


def main():
    for epoch in range(5):
        print(epoch)
        train()

if __name__ == '__main__':
    main()

