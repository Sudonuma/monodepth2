# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
from comet_ml import Experiment

import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from PIL import Image
import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

experiment = Experiment(api_key="l6NAe3ZOaMzGNsrPmy78yRnEv", project_name="monodepth2", workspace="tehad", auto_metric_logging=False)
#experiment = Experiment(api_key="l6NAe3ZOaMzGNsrPmy78yRnEv", project_name="depth2", workspace="tehad", auto_metric_logging=False)


class Trainer:
    def __init__(self, options):
        self.opt = options
        #self.experiment = experiment
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        num_val_samples = len(val_filenames)

        self.num_val_samples = num_val_samples
        self.num_train_samples = num_train_samples

        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()
        allloss = []
        thisloss = 0
        reproj_loss_per_epoch = 0
        self.val_running_loss = 0
        self.val_reproj_running_loss = 0

        for batch_idx, inputs in enumerate(self.train_loader):

            self.batch_index = inputs['target_folder']
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            # experiment.log_metric('loss before back prop', losses["loss"].cpu().detach().numpy(), epoch=self.step)
            # print('this is the loss BEFORE backprop', losses["loss"].cpu().detach().numpy())
            # print('step before backprop', self.step)


            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            # lahne sajjal el loss of each batch , amal akia fazet appendw felikhr amal average
            # print('this is the loss AFTER backprop', losses["loss"].cpu().detach().numpy())

            duration = time.time() - before_op_time
            # experiment.log_metric('loss after backprop', losses["loss"].cpu().detach().numpy(), epoch=self.step)
            # print('step after backrop', self.step)
            
            # if batch_idx==163:
            #     experiment.log_metric('loss in last batch', losses["loss"].cpu().detach().numpy(), epoch=self.epoch)

            # log less frequently after the first 2000 steps to save time & disk space
            # early_phase = batch_idx % 40 == 0 and self.step < 2000
            # late_phase = self.step % 2000 == 0
            # early_phase = batch_idx % 40 == 0
           
            # print('this is batch index',batch_idx)
            # print(early_phase)
            # print()
            # f = early_phase or late_phase
            # print('f', f)
            # if early_phase or late_phase:

                # if "depth_gt" in inputs:
                #     self.compute_depth_losses(inputs, outputs, losses)

                # self.log("train", inputs, outputs, losses)
            thisloss += losses["loss"].cpu().detach().numpy()
            reproj_loss_per_epoch += losses["reproj_loss"].cpu().detach().numpy()
            # print('accumukate',thisloss)
            allloss.append(losses["loss"].cpu().detach().numpy())
            # print('you list',allloss)
            self.val(self.val_running_loss, self.val_reproj_running_loss)
            # self.val(self.val_reproj_running_loss)
            # print('in train loop',self.val_running_loss)
            # val_running_loss =  
            # print(self.step)
            self.step += 1
            # self.log_time(batch_idx, duration, losses["loss"].cpu().data)
            # print('devide by',int(self.num_train_samples/self.opt.batch_size))
            #here they are backprogated?
        self.log_time(batch_idx, duration, losses["loss"].cpu().data)
        thisloss /= int(self.num_train_samples/self.opt.batch_size)
        reproj_loss_per_epoch /= int(self.num_train_samples/self.opt.batch_size)
        self.val_running_loss /= int(self.num_val_samples/self.opt.batch_size)
        self.val_reproj_running_loss /= int(self.num_val_samples/self.opt.batch_size)
        # print('devide by',int(self.num_val_samples/self.opt.batch_size))
        print('average validation',self.val_running_loss)
        print('average reprojection validation',self.val_reproj_running_loss)

        # print('average loss', thisloss)
        #experiment.log_metric('last batch loss', losses["loss"].cpu().detach().numpy(), epoch=self.epoch)
        experiment.log_metric('average loss druing training', thisloss, epoch=self.epoch)
        experiment.log_metric('average reprojection loss during training', reproj_loss_per_epoch, epoch=self.epoch)
        self.log("train", inputs, outputs, thisloss)
        experiment.log_metric('val loss ', self.val_running_loss, epoch=self.epoch)
        experiment.log_metric('reproj val loss ', self.val_reproj_running_loss, epoch=self.epoch)
        self.log("val", inputs, outputs, self.val_running_loss)
        for j in range(min(1, self.opt.batch_size)):
            #print('mask output to visualise',outputs["identity_selection/{}".format(0)][j][None, ...].cpu().detach().numpy().shape)
            #experiment.log_image(Image.fromarray(np.squeeze(outputs["identity_selection/{}".format(0)][j][None, ...].cpu().detach().numpy()),'L').convert('1'), name="identity_selection0")
            mask = plt.figure()
            automask = Image.fromarray(np.squeeze(outputs["identity_selection/{}".format(0)][j][None, ...].cpu().detach().numpy()))
            mask_im =mask.add_subplot(1, 1, 1, frameon = False)
            mask_im.imshow(automask, cmap = 'gist_gray')
            experiment.log_figure(figure_name="automask_0/{}".format(j))
   

            disp = plt.figure()
            disparity = normalize_image(outputs[("disp", 0)][j])
            disparity = disparity.cpu().detach().numpy()
            disparity = np.squeeze(disparity)
            disparity = Image.fromarray(disparity)
            disp_im = disp.add_subplot(1,1,1, frameon=False)
            disp_im.imshow(disparity, cmap='magma')
            experiment.log_figure(figure_name="disp_0/{}".format(j), figure=disp)


           
# self.log_time(batch_idx, duration, losses["loss"].cpu().data)
            # experiment.log_image(Image.fromarray(np.squeeze(outputs["identity_selection/{}".format(0)][j][None, ...].cpu().detach().numpy()),'L').convert('1'), name="identity_selection0")
        # for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            # for s in self.opt.scales:
            for frame_id in self.opt.frame_ids:
                #writer.add_image("color_pred_{}_{}/{}".format(frame_id, s, j), outputs[("color", frame_id, s)][j].data, self.step)

                #print('input data',inputs[("color", frame_id, 0)][j].data.size())
                #print('input data numpy shape',inputs[("color", frame_id, 0)][j].data.cpu().detach().numpy().shape)
                if frame_id != 0:
                   # writer.add_image("disp_{}/{}".format(s, j), normalize_image(outputs[("disp", s)][j]), self.step)
                    fig = plt.figure()
                    disp = plt.figure()

                    #outputs[("color", frame_id, s)][j].data
                    #print('disparity output',outputs[("disp", 0)][j])
                    warped_image = outputs[("color", frame_id, 0)][j].permute(1, 2, 0).cpu().detach().numpy()
                    disptonumpy = outputs[("disp", 0)][j].permute(1, 2, 0).cpu().detach().numpy()
                    #print(disptonumpy)
                    disparity = normalize_image(outputs[("disp", 0)][j])
                    print(disparity.size())
                    disparity = disparity.cpu().detach().numpy()
                    disparity = np.squeeze(disparity)
         
                    print('disarity',disparity.shape)
                    disparity = Image.fromarray(disparity)
                    #fig = plt.figure()
                
                    #fig.figimage(input_image)
                    im = fig.add_subplot(1,1,1, frameon=False)
                    disp_im = disp.add_subplot(1,1,1, frameon=False)
                    im.imshow(warped_image)
                    disp_im.imshow(disparity, cmap='magma')
                    experiment.log_figure(figure_name="color_pred_{}_0/{}".format(frame_id, j), figure=fig)
                    #experiment.log_figure(figure_name="disp_0/{}".format(j), figure=disp)
                #experiment.log_image(Image.fromarray(inputs[("color", frame_id, 0)][j].permute(1, 2, 0).cpu().detach().numpy(), 'RGB'), name= "color_{}_0/{}".format(frame_id, j))
                #experiment.log_image(inputs[("color", frame_id, 0)][j],image_channels='first', name= "color_{}_0/{}".format(frame_id, j))
                #experiment.log_image(Image.fromarray(np.squeeze(inputs[("color", frame_id, 0)][j].data.cpu().detach().numpy()), 'RGB'), name= "color_{}_0/{}".format(frame_id, j))
                
                # if s == 0 and frame_id != 0:
                #     experiment.log_image("color_pred_{}_{}/{}".format(frame_id, s, j), outputs[("color", frame_id, s)][j].data, self.step)

                #     writer.add_image("disp_{}/{}".format(s, j), normalize_image(outputs[("disp", s)][j]), self.step)

            # outputs[("color", frame_id, s)][j].data
            # inputs.pop('target_folder')
            plt.close('all')

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
elf.batch_index = inputs['target_folder']       """

        for key, ipt in inputs.items():
            if key != 'target_folder':
                
                inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        #also here are not backpropagated
        # experiment.log_metric('loss per batch', losses["loss"].cpu().detach().numpy())
        #print('begin////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////')
        # print("this is the output", outputs)
        #for key, value in outputs.items():
        #    print('key',key, "value size",value.size())
        #    if key == "('sample', -1, 0)":
        #        print("dict key",value.size())
        #print('end/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////')
        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self, val_running_loss, val_reproj_running_loss):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            self.val_running_loss += losses["loss"].cpu().detach().numpy()
            self.val_reproj_running_loss += losses["reproj_loss"].cpu().detach().numpy()
            # print('inside validation',self.val_running_loss)
            # if self.step % ((self.num_train_samples/self.opt.batch_size)-1) == 0:
            #     val_running_loss /= int(self.num_train_samples/self.opt.batch_size)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)], self.batch_index)
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T, self.batch_index)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        total_reproj_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojloss_alone = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                no_optimise = combined
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            reprojloss_alone += no_optimise.mean()
            loss += to_optimise.mean()

            # print('here is the loss', loss)
            # experiment.log_metric('compare loss', loss.cpu().detach().numpy())

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            reprojloss_alone += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_reproj_loss += reprojloss_alone
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        # experiment.log_metric('compare loss', loss.cpu().detach().numpy())
        total_loss /= self.num_scales
        total_reproj_loss /= self.num_scales
        losses["loss"] = total_loss
        losses["reproj_loss"] = total_reproj_loss
        
        #here they are not backpropagated just computed
        
        # experiment.log_metric('total looss in compute loss', total_loss.cpu().detach().numpy(), epoch=self.epoch)
        # experiment.log_metric('losses["loss"]', losses["loss"].cpu().detach().numpy(), epoch=self.step)
        # print('steo inside compute loss', self.step)

        # print('this also is the loss before backprop in compute loss', losses["loss"].cpu().detach().numpy())
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        #experiment.log_metrics(losses.cpu().numpy())
        #for l, v in losses.items():
        #name = mode + "average loss"
            #writer.add_scalar("{}".format(name), losses, self.step)
            # experiment.log_metric("loss in log {}".format(l), v.cpu().detach().numpy(), epoch= self.step)

        for j in range(min(1, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    #writer.add_image("color_{}_{}/{}".format(frame_id, s, j), inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image("color_pred_{}_{}/{}".format(frame_id, s, j), outputs[("color", frame_id, s)][j].data, self.step)

                        writer.add_image("disp_{}/{}".format(s, j), normalize_image(outputs[("disp", s)][j]), self.step)

                    if self.opt.predictive_mask:
                        for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                             writer.add_image("predictive_mask_{}_{}/{}".format(frame_id, s, j), outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...], self.step)

                    elif not self.opt.disable_automasking:
                        writer.add_image(
                            "automask_{}/{}".format(s, j),
                            outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
