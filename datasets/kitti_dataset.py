# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from comet_ml import Experiment

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

experiment = Experiment(api_key="l6NAe3ZOaMzGNsrPmy78yRnEv", project_name="depth2", workspace="tehad")

class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                   [0, 1.92, 0.5, 0],
        #                   [0, 0, 1, 0],
        #                   [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1242, 375)
        #focal_0 = 2323.209349931021
        #focal_13 = 1403.7527118871835
       # self.K = np.array([[focal_0/1920, 0, 0.5, 0],
         #                  [0, focal_0/1080, 0.5, 0],
          #                 [0, 0, 1, 0],
           #                [0, 0, 0, 1]], dtype=np.float32)
        f0 = 2323.209349931021
        f1 = 1403.7527118871835
        f2 = 1403.7527118871835
        f3 = 2422.4124
        f4 = 1403.7527118871835
        f5 = 1039.7880938825083
        f6 = 1403.7527118871835
        f7 = 1966.306527777393
        f8 = 1403.7527118871835
        f9 = 2016.812738295524
        f10 = 1403.7527118871835
        f11 = 2362.090502257575
        f12 = 1403.7527118871835
        f13 = 1403.7527118871835
        f14 = 1403.7527118871835
        f15 = 1403.7527118871835
        
        focals = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15]
        # self.K = np.array([[[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
              
        #       [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
              
        #      [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
              
        #       [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
 
        #      [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
             
        #       [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
              
        #       [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
              
        #       [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
              
        #       [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
              
        #       [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
              
        #       [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
              
        #       [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
              
        #       [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
              
        #       [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
              
        #       [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
              
        #       [[2218.16630365/1920, 0, 0.5, 0],
        #       [0, 2218.16630365/1080, 0.5, 0],
        #       [0, 0, 1, 0],
        #       [0, 0, 0, 1]],
             
        #      ], dtype=np.float32)

        self.K = np.array([
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
              
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
 
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
                        
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
              
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
 
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
              
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
              
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
 
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],

                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
              
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
 
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
              
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
              
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
 
                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],

                          [[0.58, 0, 0.5, 0],
                          [0, 1.92, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
             
             ], dtype=np.float32)



        self.full_res_shape = (1920, 1080)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        #self.folder  = MonoDataset.return_folder(self)
        #print(self.folder)
        
    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)
        #self.folder = KITTIDataset.folder
    
    def get_image_path(self, folder, frame_index, side):
        #f_str = "{:010d}{}".format(frame_index, self.img_ext)
        f_str = str(frame_index)+str(self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        # print(image_path)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
