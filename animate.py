import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
from scipy.spatial import ConvexHull
import numpy as np


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def animate(config, generator, kp_detector, tdmm, checkpoint, log_dir, dataset, with_eye):

    def batch_orth_proj(X, camera):
        camera = camera.clone().view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:]
        X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
        shape = X_trans.shape
        Xn = (camera[:, :, 0:1] * X_trans)
        return Xn

    log_dir = os.path.join(log_dir, 'animation')
    png_dir = os.path.join(log_dir, 'png')
    animate_params = config['animate_params']

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector, tdmm=tdmm)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if torch.cuda.is_available():
        generator = generator.cuda()
        kp_detector = kp_detector.cuda()
        tdmm = tdmm.cuda()

    generator.eval()
    kp_detector.eval()
    tdmm.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            driving_video = x['driving_video']
            source_frame = x['source_video'][:, :, 0, :, :]
            driving_name = x['driving_name'][0]
            source_name = x['source_name'][0]

            if torch.cuda.is_available():
                driving_video = driving_video.cuda()
                source_frame = source_frame.cuda()

            kp_source = kp_detector(source_frame)
            kp_driving_initial = kp_detector(driving_video[:, :, 0])
            source_codedict = tdmm.encode(source_frame)
            source_verts, source_transformed_verts, _ = tdmm.decode_flame(source_codedict)
            source_albedo = tdmm.extract_texture(source_frame, source_transformed_verts, with_eye=with_eye)
            driving_init_codedict = tdmm.encode(driving_video[:, :, 0])
            driving_init_verts, driving_init_transformed_verts, _ = tdmm.decode_flame(driving_init_codedict)

            for frame_idx in range(driving_video.shape[2]):
                driving_frame = driving_video[:, :, frame_idx]
                kp_driving = kp_detector(driving_frame)
                driving_codedict = tdmm.encode(driving_frame)

                # calculate relative 3D motion in the code space
                if relative:
                    delta_shape = source_codedict['shape'] + driving_codedict['shape'] - driving_init_codedict['shape']
                    delta_exp = source_codedict['exp'] + driving_codedict['exp'] - driving_init_codedict['exp']
                    delta_pose = source_codedict['pose'] + driving_codedict['pose'] - driving_init_codedict['pose']
                else:
                    delta_shape = source_codedict['shape']
                    delta_exp = driving_codedict['exp']
                    delta_pose = driving_codedict['pose']

                delta_source_verts, _, _ = tdmm.flame(shape_params=delta_shape,
                                                      expression_params=delta_exp,
                                                      pose_params=delta_pose)

                if relative:
                    delta_scale = source_codedict['cam'][:, 0:1] * driving_codedict['cam'][:, 0:1] / driving_init_codedict['cam'][:, 0:1]
                    delta_trans = source_codedict['cam'][:, 1:] + driving_codedict['cam'][:, 1:] - driving_init_codedict['cam'][:, 1:]
                else:
                    delta_scale = driving_codedict['cam'][:, 0:1]
                    delta_trans = driving_codedict['cam'][:, 1:]

                delta_cam = torch.cat([delta_scale, delta_trans], dim=1)
                delta_source_transformed_verts = batch_orth_proj(delta_source_verts, delta_cam)
                delta_source_transformed_verts[:, :, 1:] = - delta_source_transformed_verts[:, :, 1:]

                render_ops = tdmm.render(source_transformed_verts, delta_source_transformed_verts, source_albedo)

                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, **animate_params['normalization_params'])
                out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm, render_ops=render_ops,
                                        driving_features=driving_codedict)

                out['kp_driving'] = kp_driving
                out['kp_source'] = kp_source
                out['kp_norm'] = kp_norm

                del out['sparse_deformed']

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source_frame,
                                                                                    driving=driving_frame, out=out)
                visualizations.append(visualization)

            result_name = "-".join([x['driving_name'][0].split('.')[0], x['source_name'][0].split('.')[0]])
            if not os.path.exists(os.path.join(png_dir, result_name)):
                os.mkdir(os.path.join(png_dir, result_name))
            # save png
            for i in range(len(predictions)):
                imageio.imsave(os.path.join(png_dir, result_name + '/%07d.png' % i), (255 * predictions[i]).astype(np.uint8))
            # save gif/mp4
            image_name = result_name + animate_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

