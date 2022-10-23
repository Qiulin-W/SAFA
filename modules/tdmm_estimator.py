import os
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from skimage.io import imread
from pytorch3d.io import load_obj

from modules.util import mymobilenetv2, AntiAliasInterpolation2d
from modules.renderer_util import *
from modules.FLAME import FLAME

import pickle

flame_model_dir = './modules'
flame_config = {
    'model':{
        'n_shape': 100,
        'n_exp': 50,
        'n_pose': 6,
        'n_cam': 3,
        'uv_size': 256,
        'topology_path': os.path.join(flame_model_dir, 'data', 'head_template.obj'),
        'flame_model_path': os.path.join(flame_model_dir, 'data', 'generic_model.pkl'),
        'flame_lmk_embedding_path': os.path.join(flame_model_dir, 'data', 'landmark_embedding.npy'),
        'face_mask_path': os.path.join(flame_model_dir, 'data', 'uv_face_mask.png'),
        'face_eye_mask_path': os.path.join(flame_model_dir, 'data', 'uv_face_eye_mask.png')
    },
    'dataset':{
        'image_size': 256
    }
}

class TDMMEstimator(nn.Module):
    def __init__(self):
        super(TDMMEstimator, self).__init__()

        code_dim = flame_config['model']['n_shape'] + flame_config['model']['n_exp'] + flame_config['model']['n_pose'] + flame_config['model']['n_cam']
        self.encoder = mymobilenetv2(num_classes=code_dim, image_size=flame_config['dataset']['image_size'])
        self.flame = FLAME(flame_config['model'])

        # rasterizer
        self.rasterizer = Pytorch3dRasterizer(flame_config['dataset']['image_size'])
        self.uv_rasterizer = Pytorch3dRasterizer(flame_config['model']['uv_size'])

        # mesh template details
        verts, faces, aux = load_obj(flame_config['model']['topology_path'])
        uvcoords = aux.verts_uvs[None, ...]
        uvfaces = faces.textures_idx[None, ...]
        faces = faces.verts_idx[None, ...]
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)

        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)
        self.register_buffer('faces', faces)
        
        # face mask for rendering details
        # with eye and mouth
        mask = imread(flame_config['model']['face_eye_mask_path']).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_eye_mask = F.interpolate(mask, [flame_config['model']['uv_size'], flame_config['model']['uv_size']])
        self.register_buffer('uv_face_eye_mask', uv_face_eye_mask)

        # without eye and mouth
        mask = imread(flame_config['model']['face_mask_path']).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_mask = F.interpolate(mask, [flame_config['model']['uv_size'], flame_config['model']['uv_size']])
        self.register_buffer('uv_face_mask', uv_face_mask)

        self.image_size = flame_config['dataset']['image_size']

        self.sigmoid = nn.Sigmoid()

    def encode(self, images):   # ['shape', 'exp', 'pose', 'cam']
        code, feature = self.encoder(images)
        code_dict = {}
        code_dict['code'] = code.clone()
        code_dict['feature_vec'] = feature
        code[:, 156] += 10.0    # initial scale: 10.0 
        code_dict['shape'] = code[:, 0:100]
        code_dict['exp'] = code[:, 100:150]
        code_dict['pose'] = code[:, 150:156]
        code_dict['cam'] = code[:, 156:159]
        code_dict['cam'][:, 1:] = self.sigmoid(code_dict['cam'][:, 1:]) - 0.5
        return code_dict

    def decode_flame(self, code_dict):
        verts, landmarks_2d, _ = self.flame(shape_params=code_dict['shape'], 
                                           expression_params=code_dict['exp'], 
                                           pose_params=code_dict['pose'])
        transformed_verts = batch_orth_proj(verts, code_dict['cam'])
        transformed_verts[:, :, 1:] = - transformed_verts[:, :, 1:]

        landmarks_2d = batch_orth_proj(landmarks_2d, code_dict['cam'])[:, :, :2]
        landmarks_2d[:, :, 1:] = - landmarks_2d[:, :, 1:]

        landmarks_2d = landmarks_2d * self.image_size / 2 + self.image_size / 2

        return verts, transformed_verts, landmarks_2d

    def extract_texture(self, images, transformed_verts, with_eye=True):
        uv_pverts = self.world2uv(transformed_verts)
        uv_gt = F.grid_sample(images, uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2], mode='bilinear', align_corners=True)
        if with_eye:
            uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_eye_mask
        else:
            uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_mask
        albedo = uv_texture_gt
        return albedo

    def render(self, source_transformed_verts, driving_transformed_verts, source_albedo):
        batch_size = source_transformed_verts.shape[0]

        vert_motions = source_transformed_verts[:, :, 0:2] - driving_transformed_verts[:, :, 0:2]
        face_motions = face_vertices(vert_motions, self.faces.expand(batch_size, -1, -1))

        source_transformed_normals = vertex_normals(source_transformed_verts, self.faces.expand(batch_size, -1, -1))
        source_transformed_face_normals = face_vertices(source_transformed_normals, self.faces.expand(batch_size, -1, -1))
        driving_transformed_normals = vertex_normals(driving_transformed_verts, self.faces.expand(batch_size, -1, -1))
        driving_transformed_face_normals = face_vertices(driving_transformed_normals, self.faces.expand(batch_size, -1, -1))

        source_attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1),    # source image
                                       source_transformed_face_normals], -1)                 # source transformed face normal
        source_transformed_verts[:, :, 2] = source_transformed_verts[:, :, 2] + 10
        source_rendering = self.rasterizer(source_transformed_verts,
                                           self.faces.expand(batch_size, -1, -1),
                                           source_attributes)

        driving_attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1),    # reenact
                                       driving_transformed_face_normals,                      # driving transformed face normal
                                       face_motions], -1)
        driving_transformed_verts[:, :, 2] = driving_transformed_verts[:, :, 2] + 10
        driving_rendering = self.rasterizer(driving_transformed_verts,
                                                    self.faces.expand(batch_size, -1, -1), 
                                                    driving_attributes)
        # alpha
        source_alpha_images = source_rendering[:, -1, :, :][:, None, :, :]
        driving_alpha_images = driving_rendering[:, -1, :, :][:, None, :, :]

        # source image rendering
        source_uvcoords_images = source_rendering[:, :3, :, :]
        source_grid = (source_uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        source = F.grid_sample(source_albedo, source_grid, align_corners=False)
        source = source * source_alpha_images

        # reenact
        driving_uvcoords_images = driving_rendering[:, :3, :, :]; 
        driving_grid = (driving_uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        reenact = F.grid_sample(source_albedo, driving_grid, align_corners=False)
        reenact = reenact * driving_alpha_images

        # face region mask
        source_face_region_mask = (torch.sum(source.detach(), dim=1, keepdim=True) != 0).float()
        driving_face_region_mask = (torch.sum(reenact.detach(), dim=1, keepdim=True) != 0).float()

        # normal map
        source_transformed_normal_map = source_rendering[:, 3:6, :, :]
        driving_transformed_normal_map = driving_rendering[:, 3:6, :, :]

        # motion field
        motion_field = driving_rendering[:, 6:8, :, :]

        outputs = {
            'source': source*source_face_region_mask,
            'reenact': reenact*driving_face_region_mask,
            'source_face_mask': source_face_region_mask,
            'driving_face_mask': driving_face_region_mask,
            'source_normal_images': source_transformed_normal_map*source_face_region_mask,
            'driving_normal_images': driving_transformed_normal_map*driving_face_region_mask,
            'source_uv_images': source_uvcoords_images[:, :2, ...]*source_face_region_mask,
            'driving_uv_images': driving_uvcoords_images[:, :2, ...]*driving_face_region_mask,
            'motion_field': motion_field*driving_face_region_mask,
        }

        return outputs

    def world2uv(self, vertices):
        '''
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        '''
        batch_size = vertices.shape[0]
        face_vertices_ = face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1), 
                                         self.uvfaces.expand(batch_size, -1, -1), face_vertices_)[:, :3]
        return uv_vertices
