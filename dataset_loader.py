
import os
import numpy as np
from torch.utils.data import Dataset


class kitti_loader(Dataset):
    def __init__(self, data_dir='/root/dataset/kitti/sequences/',
                 train=True, skip_frames=200, npoints=100000):
        self.train = train
        self.data_dir = data_dir
        self.train = train
        self.skip_frames = skip_frames
        self.maxPoints = npoints

        self.reset()
        self.load_filenames()

    def reset(self):
        self.pointcloud_path = []
        self.label_path = []

        """ Reset scan members. """
        self.point = np.zeros((0, 3), dtype=np.float32)       # [m, 3]: x, y, z
        self.remission = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission
        self.label = np.zeros((0, 1), dtype=np.int32)     # [m, 1]: label

        self.proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32) # projected range image - [H,W] range (-1 is no data)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)  # unprojected range (list of depths for each point)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32) # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32) # projected remission - [H,W] intensity (-1 is no data)

        # projected index (for each pixel, what I am in the pointcloud)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)  # [H,W] index (-1 is no data)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)  # [H,W] mask

        # projection color with semantic labels
        self.proj_sem_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)

def load_filenames(self):
        if self.train:
            # seq = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
            seq = ['00']
            for seq_num in seq:
                folder_pc = os.path.join(self.data_dir, seq_num, 'velodyne')
                folder_lb = os.path.join(self.data_dir, seq_num, 'labels')

                file_pc = os.listdir(folder_pc)
                file_pc.sort(key=lambda x: str(x[:-4]))
                file_lb = os.listdir(folder_lb)
                file_lb.sort(key=lambda x: str(x[:-4]))

                for index in range(0, len(file_pc), self.skip_frames):
                    self.pointcloud_path.append('%s/%s' % (folder_pc, file_pc[index]))
                    self.label_path.append('%s/%s' % (folder_lb, file_lb[index]))
        else:
            seq = '08'
            folder_pc = os.path.join(self.data_dir, seq, 'velodyne')
            folder_lb = os.path.join(self.data_dir, seq, 'labels')
            file_pc = os.listdir(folder_pc)
            file_pc.sort(key=lambda x: str(x[:-4]))
            file_lb = os.listdir(folder_lb)
            file_lb.sort(key=lambda x: str(x[:-4]))
            for index in range(0, len(file_pc), self.skip_frames):
                self.pointcloud_path.append('%s/%s' % (folder_pc, file_pc[index]))
                self.label_path.append('%s/%s' % (folder_lb, file_lb[index]))

    def get_data(self, pointcloud_path, label_path):
        # points = np.load(pointcloud_path) # for npy files
        data = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
        self.point = data[:, 0:3]
        self.remission = data[:, 3]
        self.label = np.fromfile(label_path, dtype=np.float32)

    def limitDataset(self, xlim, ylim, zlim):
        # square
        # point = np.array([x for x in point if 0 < x[0] + 51.2 < 102.4 and 0 < x[1] + 51.2 < 102.4 and 0< x[2]+5 < 8])

        self.point = np.array([x for x in self.point if
                          0 < x[0] - xlim[0] < xlim[1] - xlim[0] and 0 < x[1] - ylim[0] < ylim[1] - ylim[0] and 0 < x[
                              2] - zlim[0] < zlim[1] - zlim[0]])

        if len(self.point) >= self.maxPoints:
            choice = np.random.choice(len(self.point), self.maxPoints, replace=False)
        else:
            choice = np.random.choice(len(self.point), self.maxPoints, replace=True)
        self.point = self.point[choice]
        self.label = self.label[choice]

    def do_range_projection(self,fov_up,fov_down,proj_W,proj_H):
        # laser parameters
        fov_up = fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.point, 2, axis=1)

        # get scan components
        scan_x = self.point[:, 0]
        scan_y = self.point[:, 1]
        scan_z = self.point[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= proj_W  # in [0.0, W]
        proj_y *= proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)

        # only map colors to labels that exist
        mask = self.proj_idx >= 0

        # semantics
        self.proj_sem_label[mask] = self.label[self.proj_idx[mask]]



    def __getitem__(self, index):
        self.get_data(self.pointcloud_path[index], self.label_path[index])
        self.limitDataset([-51.2, 51.2], [-51.2, 51.2], [-5, 3])
        self.do_range_projection(self,3,-25,1024,64)

        return self.proj_remission, self.proj_sem_label