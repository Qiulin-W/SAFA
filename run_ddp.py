import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset, ImageDataset

from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector
from modules.tdmm_estimator import TDMMEstimator

import torch

from train_ddp import train, train_tdmm
from reconstruction import reconstruction
from animate import animate

import torch.distributed as dist


if __name__ == "__main__":

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--with_eye", action="store_true", help="use eye part for extracting texture")
    parser.add_argument("--mode", default="train", choices=["train", "train_tdmm", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint of the whole model to restore")
    parser.add_argument("--tdmm_checkpoint", default=None, help="path to checkpoint of the tdmm estimator model to restore")

    opt = parser.parse_args()

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    if opt.mode == 'train' or opt.mode == 'train_tdmm':
        local_rank = opt.local_rank
        print("local rank: ", local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')

        if local_rank == 0:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
                copy(opt.config, log_dir)

    with open(opt.config) as f:
        config = yaml.safe_load(f)

    if opt.mode == 'train_tdmm':
        tdmm = TDMMEstimator()

        dataset = ImageDataset(data_dir=config['dataset_params']['root_dir'], 
                               meta_dir=config['dataset_params']['meta_dir'],
                               augmentation_params=config['dataset_params']['augmentation_params'])
    else:
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

        discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                                **config['model_params']['common_params'])

        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                **config['model_params']['common_params'])

        tdmm = TDMMEstimator()

        dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, kp_detector, tdmm, log_dir, dataset, local_rank, 
              with_eye=opt.with_eye, checkpoint=opt.checkpoint, tdmm_checkpoint=opt.tdmm_checkpoint)
    elif opt.mode == 'train_tdmm':
        print("Training tdmm ...")
        train_tdmm(config, tdmm, log_dir, dataset, local_rank, tdmm_checkpoint=opt.tdmm_checkpoint)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, kp_detector, tdmm, 
                       opt.checkpoint, log_dir, dataset, with_eye=opt.with_eye)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator, kp_detector, tdmm, 
                opt.checkpoint, log_dir, dataset, with_eye=opt.with_eye)

