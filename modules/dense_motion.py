from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from all sparse motions including 3D face motion and 2D affine motions.
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 2) * (num_channels + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 2, kernel_size=(7, 7), padding=(3, 3))

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
            self.occlusion1 = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
            self.down_motion = AntiAliasInterpolation2d(2, self.scale_factor)

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        heatmaps for the 2D affine motions
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        """
        T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)
        if 'jacobian' in kp_driving:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 2)

        # adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        deformed source images
        """
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions, align_corners=True)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def forward(self, source_image, kp_driving=None, kp_source=None, render_ops=None):
        if type(render_ops) != type(None):
            reenact = render_ops['reenact']
            motion_field = render_ops['motion_field']
            source_normal_map = render_ops['source_normal_images']
            driving_normal_map = render_ops['driving_normal_images']

        if self.scale_factor != 1:
            source_image = self.down(source_image)
            if type(render_ops) != type(None):
                reenact = self.down(reenact)
                motion_field = self.down_motion(motion_field)
                source_normal_map = self.down(source_normal_map)
                driving_normal_map = self.down(driving_normal_map)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source

        # REMEMBER to add identity_grid to the motion field
        identity_grid = make_coordinate_grid((h, w), type=motion_field.type())
        identity_grid = identity_grid.unsqueeze(0).repeat(bs, 1, 1, 1).permute(0, 3, 1, 2)
        motion_field = motion_field + identity_grid

        if type(render_ops) != type(None):
            reenact = reenact.unsqueeze(1)
            source_normal_map = source_normal_map.unsqueeze(1)
            driving_normal_map = driving_normal_map.unsqueeze(1)
            heatmap_representation = torch.cat([heatmap_representation,
                                                -(driving_normal_map[:, :, 2:, :, :] - source_normal_map[:, :, 2:, :, :])],
                                                dim=1)
            deformed_source = torch.cat([deformed_source,
                                        reenact],
                                        dim=1)
            out_dict['motion_field'] = motion_field

        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        input = input.view(bs, -1, h, w)
        prediction = self.hourglass(input)
        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)

        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        sparse_motion = torch.cat([sparse_motion, motion_field.unsqueeze(1)], dim=1)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation

        if self.occlusion:
            occlusion_map1 = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map1'] = occlusion_map1
            occlusion_map2 = torch.sigmoid(self.occlusion1(prediction))
            out_dict['occlusion_map2'] = occlusion_map2
        return out_dict
