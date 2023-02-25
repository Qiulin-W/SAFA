import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
from imageio import mimread
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage import io
import torch
import torch.nn.functional as F

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from modules.tdmm_estimator import TDMMEstimator

from animate import normalize_kp
from scipy.spatial import ConvexHull

import time
import cv2
import face_alignment
import glob


def bbox_increase(bboxes, img_w, img_h, ratio=1.5):
    new_bboxes = bboxes.copy().astype(float)

    w = (new_bboxes[2] - new_bboxes[0])
    h = (new_bboxes[3] - new_bboxes[1])

    new_bboxes[1] = np.where(w > h, new_bboxes[1] - (w - h) / 2, new_bboxes[1])
    new_bboxes[3] = np.where(w > h, new_bboxes[3] + (w - h) / 2, new_bboxes[3])

    new_bboxes[0] = np.where(w < h, new_bboxes[0] - (h - w) / 2, new_bboxes[0])
    new_bboxes[2] = np.where(w < h, new_bboxes[2] + (h - w) / 2, new_bboxes[2])

    center = np.stack(((new_bboxes[0] + new_bboxes[2])/2, (new_bboxes[1] + new_bboxes[3])/2), axis=0)
    img_size = np.where(w > h, w, h)
    img_size = img_size * ratio

    new_bboxes[0] = np.maximum((center[0] - img_size / 2.0), 0)
    new_bboxes[1] = np.maximum((center[1] - img_size / 2.0), 0)
    new_bboxes[2] = np.minimum(img_w, (center[0] + img_size / 2.0))
    new_bboxes[3] = np.minimum(img_h, (center[1] + img_size / 2.0))

    return new_bboxes

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def partial_state_dict_load(module, state_dict):
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

def load_checkpoints(blend_scale, config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # init generator
    generator = OcclusionAwareGenerator(blend_scale=blend_scale,
                                        **config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    # init kp_detector
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    # init tdmm estimator
    tdmm = TDMMEstimator()

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    partial_state_dict_load(generator, checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    tdmm.load_state_dict(checkpoint['tdmm'])

    if not cpu:
        generator = generator.cuda()
        kp_detector = kp_detector.cuda()
        tdmm = tdmm.cuda()

    generator.eval()
    kp_detector.eval()
    tdmm.eval()
    
    return generator, kp_detector, tdmm

def load_face_parser(cpu=False):
    from face_parsing.model import BiSeNet

    face_parser = BiSeNet(n_classes=19)
    face_parser.load_state_dict(torch.load('face_parsing/cp/79999_iter.pth', map_location=torch.device('cpu')), strict=False)
 
    if not cpu:
       face_parser.cuda()

    face_parser.eval()

    mean = torch.Tensor(np.array([0.485, 0.456, 0.406], dtype=np.float32)).view(1, 3, 1, 1)
    std = torch.Tensor(np.array([0.229, 0.224, 0.225], dtype=np.float32)).view(1, 3, 1, 1)

    if not cpu:
        face_parser.mean = mean.cuda()
        face_parser.std = std.cuda()
    else:
        face_parser.mean = mean
        face_parser.std = std

    return face_parser


def faceswap(opt, fa, generator, kp_detector, tdmm):
    source_image = np.array(io.imread(opt.source_image_pth))
    if source_image.shape[2] == 4:
        source_image = source_image[..., 0:3]
    driving_video = np.array(mimread(opt.driving_video_pth, memtest=False))

    if opt.use_detection:
        detection = fa.face_detector.detect_from_image(source_image)
        if len(detection) == 0:
            raise ValueError('No faces detected in source images')
        source_bbox = bbox_increase(detection[0][0:4], source_image.shape[1], source_image.shape[0])
        source_bbox = source_bbox.astype(np.int32)
        source = torch.tensor(source_image[np.newaxis, source_bbox[1]:source_bbox[3], source_bbox[0]:source_bbox[2]].astype(np.float32)).permute(0, 3, 1, 2)
    else:
        source = torch.tensor(source_image[np.newaxis, ...].astype(np.float32)).permute(0, 3, 1, 2)

    source = source / 255.0
    if not opt.cpu:
        source = source.cuda()
    source = F.interpolate(source, size=(256, 256), mode='bilinear', align_corners=True)
    kp_source = kp_detector(source)

    with torch.no_grad():
        predictions = []
        for i in tqdm(range(driving_video.shape[0])):
            driving_image = driving_video[i]

            if opt.use_detection:
                detection = fa.face_detector.detect_from_image(driving_image)
                if len(detection) == 0:
                    raise ValueError('No faces detected in source images')
                driving_bbox = bbox_increase(detection[0][0:4], driving_image.shape[1], driving_image.shape[0])
                driving_bbox = driving_bbox.astype(np.int32)
                driving = torch.tensor(driving_image[np.newaxis, driving_bbox[1]:driving_bbox[3], driving_bbox[0]:driving_bbox[2]].astype(np.float32)).permute(0, 3, 1, 2)
            else:
                driving = torch.tensor(driving_image[np.newaxis, ...].astype(np.float32)).permute(0, 3, 1, 2)

            driving = driving / 255.0
            if not opt.cpu:
                driving = driving.cuda()
            driving = F.interpolate(driving, size=(256, 256), mode='bilinear', align_corners=True)
            kp_driving = kp_detector(driving)

            # for face swap
            if face_parser is not None:
                blend_mask = F.interpolate(driving, size=(512, 512))
                blend_mask = (blend_mask - face_parser.mean) / face_parser.std
                blend_mask = face_parser(blend_mask)
                blend_mask = torch.softmax(blend_mask[0], dim=1)
            else:
                blend_mask = None

            blend_mask = blend_mask[:, opt.swap_index].sum(dim=1, keepdim=True)
            if opt.hard:
                blend_mask = (blend_mask > 0.5).type(blend_mask.type())

            # 3DMM rendering
            source_codedict = tdmm.encode(source)
            driving_codedict = tdmm.encode(driving)
            source_verts, source_transformed_verts, source_ldmk_2d = tdmm.decode_flame(source_codedict)
            driving_verts, driving_transformed_verts, driving_ldmk_2d = tdmm.decode_flame(driving_codedict)
            source_albedo = tdmm.extract_texture(source, source_transformed_verts, with_eye=opt.with_eye)
            render_ops = tdmm.render(source_transformed_verts, driving_transformed_verts, source_albedo)

            out = generator(source, kp_source=kp_source, kp_driving=kp_driving, render_ops=render_ops, 
                            blend_mask=blend_mask, driving_image=driving, driving_features=driving_codedict)

            if opt.use_detection:
                # fit to the original video
                bbox_w, bbox_h = driving_bbox[2]-driving_bbox[0], driving_bbox[3]-driving_bbox[1]
                prediction = F.interpolate(out['prediction'], size=(bbox_h, bbox_w), mode='bilinear', align_corners=True)
                driving_image[driving_bbox[1]:driving_bbox[3], driving_bbox[0]:driving_bbox[2], :] = np.transpose(prediction.data.cpu().numpy(), [0, 2, 3, 1])[0] * 255.0
                predictions.append(driving_image)
            else:
                predictions.append((np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255.0).astype(np.uint8))

    imageio.mimsave(opt.result_video_pth, predictions, fps=25)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='', help="path to checkpoint to restore")
    parser.add_argument("--with_eye", action="store_true", help="use eye part for extracting texture")
    parser.add_argument("--source_image_pth", default='', help="path to source image")
    parser.add_argument("--driving_video_pth", default='', help="path to driving video")
    parser.add_argument("--result_video_pth", default='result.mp4', help="path to output")

    parser.add_argument("--swap_index", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", type=lambda x: list(map(int, x.split(','))),
                        help='index of swaped parts')
    parser.add_argument("--hard", action="store_true", help="use hard segmentation labels for blending")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--use_detection", action="store_true", help="use detected bbox")

    opt = parser.parse_args()

    blend_scale = (256 / 4) / 512
    generator, kp_detector, tdmm = load_checkpoints(blend_scale=blend_scale, config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    face_parser = load_face_parser(opt.cpu)
    print("face_parser is loaded!")

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='sfd', 
                                      device='cpu' if opt.cpu else 'cuda')
    faceswap(opt, fa, generator, kp_detector, tdmm)
            