# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataset_loader import kitti_loader
from torch.utils.data import DataLoader
from module.network import U_Net
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torchsummary import summary


# dataloader = DataLoader(dataset, batch_size=cfg.batch_size * cfg.num_gpus, shuffle=True,
#                         num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
# test_dataset = kitti_loader(data_dir=cfg.root_dir, point_cloud_files=cfg.point_cloud_files,
#                             data_type=args.dataset_type, labels_files=cfg.labels_files,
#                             train=False)
# test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size * cfg.num_gpus, shuffle=False,
#                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
# dt = np.dtype(['x','y','z','i'])
# def plot2d(twod_array,title):
#     plt.figure(figsize=(15, 2))
#     plt.title(title)
#     plt.imshow(twod_array)
#     plt.colorbar()
#     plt.show()

# dataset = kitti_loader()
# scan,_ =dataset.__getitem__(index=0)
# print("The length of dataset is = ", len(dataset))
# proj_remission, proj_sem_label = dataset[1]
#
# print("1.....", np.shape(proj_remission))
# print("2.....", np.shape(proj_sem_label))

# plot2d(proj_remission, "proj_remission")
# plot2d(proj_sem_label,"proj_sem_label")
#
# # Masking cars only in 2D
# proj_label = np.zeros((64, 1024), dtype=np.float32)
# proj_label[proj_sem_label == 10] = 1
# print(proj_label[18, 0:200])
#
# plot2d(proj_label,"proj_label","Greys")


dataset = kitti_loader(data_dir="/home/dong/dataset/sequences/",train=True, skip_frames=5, npoints=100000)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True)
#
in_ch = 1
out_ch = 2 #no of classes 2 in our case (car, environment)

model= U_Net(in_ch, out_ch)
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
summary(model, input_size=(1, 64, 1024))
# print(model)

step_losses = []
epoch_losses = []

def main():
    epochs = 5
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for batch_idx, (proj_remission, proj_sem_label) in enumerate(dataloader):
            # print(proj_remission.shape)
            # print("Epoch == ", epoch, "Batch == ", batch_idx)
            # proj_remission, proj_sem_label = proj_remission.cuda(), proj_sem_label.cuda()
            proj_remission = proj_remission.cuda()
            # proj_sem_label = torch.tensor(proj_sem_label, dtype=torch.long, device=torch.device('cuda'))
            proj_sem_label = proj_sem_label.cuda(non_blocking=True).long()
            # print("proj_sem_label shape", proj_sem_label.size())
            optimizer.zero_grad()
            out = model(proj_remission)
            loss = loss_crs(out, proj_sem_label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_losses.append(loss.item())
        epoch_losses.append(epoch_loss / len(dataloader))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(step_losses)
    axes[1].plot(epoch_losses)

    model_name = "Range_Net_Test.pth"
    torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    main()
    print("done")

