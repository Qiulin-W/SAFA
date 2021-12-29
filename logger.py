import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
import cv2
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections

from optic_flow_utils import *

from modules.util import make_coordinate_grid


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if visualizer_params:
            if not os.path.exists(self.visualizations_dir):
                os.makedirs(self.visualizations_dir)
            self.visualizer = Visualizer(**visualizer_params)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)
        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None, tdmm=None,
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None, optimizer_tdmm=None,
                 local_rank=None):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
            generator.to(local_rank)
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
            kp_detector.to(local_rank)

        if tdmm is not None:
            tdmm.load_state_dict(checkpoint['tdmm'])
            tdmm.to(local_rank)

        if discriminator is not None:
            try:
                discriminator.load_state_dict(checkpoint['discriminator'])
                discriminator.to(local_rank)
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])

        if optimizer_tdmm is not None:
            optimizer_tdmm.load_state_dict(checkpoint['optimizer_tdmm'])

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)

    def log_epoch_tdmm(self, epoch, models):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)

class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []

        # Source image with keypoints
        source = source.data.cpu()
        kp_source = out['kp_source']['value'].data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Equivariance visualization
        if 'transformed_frame' in out:
            transformed = out['transformed_frame'].data.cpu().numpy()
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = out['transformed_kp']['value'].data.cpu().numpy()
            images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        # Deformed image
        if 'deformed' in out:
            deformed = out['deformed'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['value'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)

        # Occlusion map
        if 'occlusion_map1' in out:
            occlusion_map = out['occlusion_map1'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)
        
        if 'occlusion_map2' in out:
            occlusion_map1 = 1.0 - out['occlusion_map2'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map1 = F.interpolate(occlusion_map1, size=source.shape[1:3]).numpy()
            occlusion_map1 = np.transpose(occlusion_map1, [0, 2, 3, 1])
            images.append(occlusion_map1)

        # Deformed images according to each individual transform
        if 'sparse_deformed' in out:
            full_mask = []
            for i in range(out['sparse_deformed'].shape[1]):
                image = out['sparse_deformed'][:, i].data.cpu()
                image = F.interpolate(image, size=source.shape[1:3])
                mask = out['mask'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
                mask = F.interpolate(mask, size=source.shape[1:3])
                image = np.transpose(image.numpy(), (0, 2, 3, 1))
                mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['sparse_deformed'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))

                images.append(image)
                if i != 0:
                    images.append(mask * color)
                else:
                    images.append(mask)

                full_mask.append(mask * color)
            images.append(sum(full_mask))

        if 'reenact' in out:
            image = out['reenact'].data.cpu()
            image = np.transpose(image.numpy(), (0, 2, 3, 1))
            images.append(image)

        if 'mask' in out:
            full_mask = []
            full_mask_bin = []

            mask_bin = F.interpolate(out['mask'], size=source.shape[1:3], mode='bilinear')
            mask_bin = (torch.max(mask_bin, dim=1, keepdim=True)[0] == mask_bin).float()
            tdmm_mask = None

            # formulate mask bin
            for i in range(out['mask'].shape[1]):
                mask = out['mask'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
                mask = F.interpolate(mask, size=source.shape[1:3], mode='bilinear')
                mask = np.transpose(mask.numpy(), (0, 2, 3, 1))
                mask_bin_part = mask_bin[:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
                mask_bin_part = np.transpose(mask_bin_part.numpy(), (0, 2, 3, 1))

                if i != 0:
                    if i != (out['mask'].shape[1] - 1):
                        color = np.array(self.colormap((i - 1) / (out['mask'].shape[1] - 1)))[:3]
                    else:
                        tdmm_mask = mask_bin_part
                        color = np.array((1.0, 1.0, 1.0))    # full white for 3D mask
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))

                full_mask.append(mask * color)
                full_mask_bin.append(mask_bin_part * color)

            images.append(sum(full_mask))
            images.append(0.3 * driving + 0.7 * sum(full_mask))
            images.append(sum(full_mask_bin))
            images.append(0.3 * driving + 0.7 * sum(full_mask_bin))
            images.append(tdmm_mask)

        identity_grid = make_coordinate_grid((source.shape[1], source.shape[2]), type=source.type())
        identity_grid = identity_grid.data.cpu().numpy()

        if 'motion_field' in out and out['motion_field'].shape[0] == 1:
            motion_field = F.interpolate(out['motion_field'], size=source.shape[1:3], mode='bilinear')
            motion_field = motion_field.squeeze().permute(1, 2, 0).data.cpu().numpy()
            motion_field = motion_field - identity_grid
            optic_flow = flow_to_image(motion_field)[None, ...]
            images.append(optic_flow)
        
        if 'deformation' in out and out['deformation'].shape[0] == 1:
            deformation = F.interpolate(out['deformation'].permute(0, 3, 1, 2), size=source.shape[1:3], mode='bilinear')
            deformation = deformation.squeeze().permute(1, 2, 0).data.cpu().numpy()
            deformation = deformation - identity_grid
            optic_flow = flow_to_image(deformation)[None, ...]
            images.append(optic_flow)
        
        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
