import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, GADEUpBlock2d, ContextualAttention
from modules.dense_motion import DenseMotionNetwork
from modules.util import AntiAliasInterpolation2d


class OcclusionAwareGenerator(nn.Module):
    """
    Given source image, dense motion field, occlusion map, and driving 3DMM code, 
    the U-net shaped Generator outputs the final reconstructed image.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False,
                 blend_scale=1, num_dilation_group=4, dilation_rates=[1,2,4,8]):
        super(OcclusionAwareGenerator, self).__init__()
        print("blend_scale: ", blend_scale)

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            #up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            # We can also inject the driving 3DMM code to the upblock, see GADEUpBlock2d in util.py for details.
            up_blocks.append(GADEUpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))

        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

        if blend_scale == 1:
            self.blend_downsample = lambda x: x
        else:
            self.blend_downsample = AntiAliasInterpolation2d(1, blend_scale)

        self.fc_1 = nn.Linear(1280, in_features)
        self.fc_2 = nn.Linear(1280, in_features)

        self.context_atten_layer = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True)

        self.num_dilation_group = num_dilation_group
        for i in range(num_dilation_group):
            self.__setattr__('conv{}'.format(str(i).zfill(2)), nn.Sequential(
            nn.Conv2d(in_features, in_features//num_dilation_group, kernel_size=3, dilation=dilation_rates[i], padding=dilation_rates[i]),
            nn.BatchNorm2d(in_features//num_dilation_group, affine=True),
            nn.ReLU())
            )

        self.concat_conv = nn.Sequential(
                nn.Conv2d(in_features*2, in_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_features, affine=True),
                nn.ReLU()
                )

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving=None, kp_source=None, render_ops=None, 
                      blend_mask=None, driving_image=None, driving_features=None):

        # Encoder
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_source=kp_source, kp_driving=kp_driving, 
                                                     render_ops=render_ops)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']
            output_dict['motion_field'] = dense_motion['motion_field']
            output_dict['reenact'] = render_ops['reenact']

            if 'occlusion_map1' in dense_motion:
                occlusion_map1 = dense_motion['occlusion_map1']
                output_dict['occlusion_map1'] = occlusion_map1
                occlusion_map2 = dense_motion['occlusion_map2']
                output_dict['occlusion_map2'] = occlusion_map2
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']

            out = self.deform_input(out, deformation)

            if occlusion_map1 is not None:
                if out.shape[2] != occlusion_map1.shape[2] or out.shape[3] != occlusion_map1.shape[3]:
                    occlusion_map1 = F.interpolate(occlusion_map1, size=out.shape[2:], mode='bilinear')
                    occlusion_map2 = F.interpolate(occlusion_map2, size=out.shape[2:], mode='bilinear')
                output_dict['occlusion_map1'] = occlusion_map1
                output_dict['occlusion_map2'] = occlusion_map2

            output_dict['deformation'] = deformation
            output_dict["deformed"] = self.deform_input(source_image, deformation)

        # Geometrically-Adaptive Denormalization (GADE) Layer
        code = driving_features['feature_vec']

        r = self.fc_1(code).unsqueeze(-1).unsqueeze(-1).expand_as(out)
        beta = self.fc_2(code).unsqueeze(-1).unsqueeze(-1).expand_as(out)
        addin_driving = r * out + beta
        out = (1.0 - occlusion_map1) * addin_driving + occlusion_map1 * out

        if blend_mask is not None:
            blend_mask = self.blend_downsample(blend_mask)
            enc_driving = self.first(driving_image)
            for i in range(len(self.down_blocks)):
                enc_driving = self.down_blocks[i](enc_driving)
            out = enc_driving * (1 - blend_mask) + out * blend_mask

        # Bottleneck block
        out = self.bottleneck(out)

        # Contextual Attention (CA) Module
        ca_occ = occlusion_map2
        # 1. CA branch
        ca_out = self.context_atten_layer(out, out, 1.0 - ca_occ)
        # 2. Dilation pyramid branch
        tmp = []
        for i in range(self.num_dilation_group):
            tmp.append(self.__getattr__('conv{}'.format(str(i).zfill(2)))(out))
        dilated_out = torch.cat(tmp, dim=1)
        # 3. concat and fuse
        out = torch.cat([ca_out, dilated_out], dim=1)
        out = self.concat_conv(out)

        for i in range(len(self.up_blocks)):
            #out = self.up_blocks[i](out)
            out = self.up_blocks[i](out, code)

        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict

