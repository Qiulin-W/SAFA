import cv2
import torch
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from modules.tdmm_estimator import TDMMEstimator
from torch.utils.data import DataLoader
from frames_dataset import FramesDataset, ImageDataset
from tqdm import tqdm


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="", help="directory containing images to inference")
    parser.add_argument("--gpu", action="store_true", help="run the inference on gpu")
    parser.add_argument("--with_eye", action="store_true", help="use eye part for extracting texture")
    parser.add_argument("--tdmm_checkpoint", default=None, help="path to checkpoint of the tdmm estimator model to restore")
    
    opt = parser.parse_args()

    checkpoint = torch.load(opt.tdmm_checkpoint, map_location=torch.device('cpu'))

    tdmm = TDMMEstimator()
    tdmm.load_state_dict(checkpoint['tdmm'], strict=False)
    if opt.gpu:
        tdmm = tdmm.cuda()

    dataset = ImageDataset(data_dir=opt.data_dir, meta_dir=None, augmentation_params=None)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)
    tdmm.eval()

    for i, x in tqdm(enumerate(dataloader)):

        if opt.gpu:
            x['image'] = x['image'].cuda()

        codedict = tdmm.encode(x['image'])
        verts, transformed_verts, landmark_2d = tdmm.decode_flame(codedict)

        # extract albedo and rendering
        albedo = tdmm.extract_texture(x['image'], transformed_verts, with_eye=opt.with_eye)
        outputs = tdmm.render(transformed_verts, transformed_verts, albedo)

        image = x['image'].squeeze().permute(1, 2, 0)
        source = outputs['source'].squeeze().permute(1, 2, 0).detach()
        normal_map = outputs['source_normal_images'].squeeze().permute(1, 2, 0).detach()

        if opt.gpu:
            image = image.cpu()
            normal_map = normal_map.cpu()
            source = source.cpu()

        image = image.numpy()
        source = source.numpy()
        normal_map = normal_map.numpy()

        cv2.imshow('input', image[..., ::-1])
        cv2.imshow('source', source[..., ::-1])
        cv2.imshow('normal', normal_map)
        cv2.waitKey(0)