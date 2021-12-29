import os
import cv2
import pickle
import numpy as np
from imageio import mimread
from tqdm import tqdm
import face_alignment
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--video_dir", default='', help="video directory")
    parser.add_argument("--out_dir", default='video_ldmk_meta', help="directory to save output pickle files")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="use cpu mode")
    parser.add_argument("--vis_ldmk", dest="vis_ldmk", action="store_true", help="visualize predicted landmarks")

    opt = parser.parse_args()

    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)

    if opt.cpu:
        device = 'cpu'
    else:
        device = 'cuda'

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='sfd', device=device)

    for video_name in tqdm(os.listdir(opt.video_dir)):
        print('processing {}'.format(video_name))
        video_pth = os.path.join(opt.video_dir, video_name)
        video = np.array(mimread(video_pth))
        
        video_ldmk_meta = {}
        for i in range(video.shape[0]):
            ldmk_pred = fa.get_landmarks(video[i])[0]
            video_ldmk_meta[i] = {}
            if len(ldmk_pred) == 0:
                continue
            else:
                video_ldmk_meta[i]['ldmk'] = ldmk_pred
                if opt.vis_ldmk:
                    img = video[i]
                    for j in range(ldmk_pred.shape[0]):
                        cv2.circle(img, (int(ldmk_pred[j, 0]), int(ldmk_pred[j, 1])), radius=1, color=(255, 255, 255))
                    cv2.imshow('vis', img)
                    cv2.waitKey(0)

        f = open(os.path.join(opt.out_dir, video_name.split('.')[0] + ".pkl"), 'wb')
        pickle.dump(video_ldmk_meta, f)
        f.close()

