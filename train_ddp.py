from tqdm import trange, tqdm
import torch

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel, TdmmFullModel

from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import DataLoader
from frames_dataset import DatasetRepeater

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def train(config, generator, discriminator, kp_detector, tdmm, 
          log_dir, dataset, local_rank, with_eye=True, checkpoint=None, tdmm_checkpoint=None):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_tdmm = torch.optim.Adam(tdmm.parameters(), lr=train_params['lr_tdmm'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector,
                                      local_rank)
    else:
        start_epoch = 0
        tdmm_checkpoint = torch.load(tdmm_checkpoint, map_location=torch.device('cpu'))
        tdmm.load_state_dict(tdmm_checkpoint['tdmm'], strict=False)

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_tdmm = MultiStepLR(optimizer_tdmm, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_tdmm'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], num_workers=4, sampler=train_sampler)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, tdmm, train_params, with_eye=with_eye)
    generator_full = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator_full)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator_full)

    if torch.cuda.is_available():
        generator_full.to(local_rank)
        discriminator_full.to(local_rank)
        generator_full = DDP(generator_full, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        discriminator_full = DDP(discriminator_full, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # fix bn layers of pretrained tdmm model 
    generator_full._module_copies[0].tdmm.apply(fix_bn)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):

            dataloader.sampler.set_epoch(epoch)

            for x in tqdm(dataloader):
                x['source'] = x['source'].to(local_rank)
                x['driving'] = x['driving'].to(local_rank)
                x['source_ldmk_2d'] = x['source_ldmk_2d'].to(local_rank)
                x['driving_ldmk_2d'] = x['driving_ldmk_2d'].to(local_rank)

                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()
                optimizer_tdmm.step()
                optimizer_tdmm.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            scheduler_tdmm.step()

            if dist.get_rank() == 0:
                logger.log_epoch(epoch, {'generator': generator,
                                        'discriminator': discriminator,
                                        'kp_detector': kp_detector,
                                        'tdmm': tdmm,
                                        'optimizer_generator': optimizer_generator,
                                        'optimizer_discriminator': optimizer_discriminator,
                                        'optimizer_kp_detector': optimizer_kp_detector,
                                        'optimizer_tdmm': optimizer_tdmm}, inp=x, out=generated)


def train_tdmm(config, tdmm, log_dir, dataset, local_rank, tdmm_checkpoint=None):
    train_params = config['train_params']
    optimizer_tdmm = torch.optim.Adam(tdmm.parameters(), lr=train_params['lr_tdmm'], betas=(0.9, 0.999))

    if tdmm_checkpoint is not None:
        start_epoch = Logger.load_cpk(tdmm_checkpoint, tdmm=tdmm, optimizer_tdmm=optimizer_tdmm, local_rank=local_rank)
    else:
        start_epoch = 0

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], num_workers=4, sampler=train_sampler)

    tdmm_full = TdmmFullModel(tdmm)
    tdmm_full = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tdmm_full)

    if torch.cuda.is_available():
        tdmm_full.to(local_rank)
        tdmm_full = DDP(tdmm_full, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    logger = Logger(log_dir, checkpoint_freq=train_params['checkpoint_freq'])

    for epoch in trange(start_epoch, train_params['num_epochs']):
        dataloader.sampler.set_epoch(epoch)
        for i, x in tqdm(enumerate(dataloader)):
            optimizer_tdmm.zero_grad()
            x['image'] = x['image'].to(local_rank)
            x['ldmk'] = x['ldmk'].to(local_rank)

            losses_tdmm = tdmm_full(x)

            loss_values = [val for val in losses_tdmm.values()]
            loss = sum(loss_values)

            if i % 10 == 0:
                print('batch ldmk loss: ', loss)

            loss.backward()
            optimizer_tdmm.step()

            losses = {key: value.data for key, value in losses_tdmm.items()}
            logger.log_iter(losses=losses)

        if dist.get_rank() == 0:
            logger.log_epoch_tdmm(epoch, {'tdmm': tdmm, 'optimizer_tdmm': optimizer_tdmm})
