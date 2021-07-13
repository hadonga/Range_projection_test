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

dataset = kitti_loader()
# scan,_ =dataset.__getitem__(index=1)
dataloader = DataLoader(dataset,batch_size=1, shuffle= True, num_workers= 4,
                        pin_memory= True, drop_last=True)

model= U_Net()
print("Model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

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
        print(proj_remission.shape(),"    ",proj_sem_label.shape())
        # # convert it into x,y,z coordinates and i
        # x = scan[0,:, 0]  # get x
        # y = scan[0,:, 1]  # get x
        # z = scan[0,:, 2]  # get x
        # i = scan[0,:, 3]  # get intensity
        # label= labels[0,:]
        # for ii in range(len(label)):
        #     if label[ii] == "10":
        #         label[ii] = 1
        #     else:
        #         label[ii] = 0
        # img,lab = projection_points(x, y, z, i, label)
    # optimizer.zero_grad()
    # out=model(img)
    # loss=loss_crs(out,lab)
    #
    # loss.backward()
    # optimizer.step()


def main():
    for epoch in range(5):
        print(epoch)
        train()

if __name__ == '__main__':
    main()

