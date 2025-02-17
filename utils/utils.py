# import argparse
# import numpy as np
# import scipy as sp
# import scipy.stats as stats
# import matplotlib.pyplot as plt
# from scipy.stats import t
# import cv2
# import os
# import datetime

# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')    
        
        
        
# def get_colors():    
#     RGB_tuples                   = np.vstack([np.loadtxt("utils/colors.txt", skiprows=1) , np.random.uniform(0, 255, size=(10000, 3))])
#     b                            = np.where(RGB_tuples==0)
#     RGB_tuples[b]                = 1
    
#     return RGB_tuples


# def get_prediction_interval(y, y_hat, x, x_hat):
#     n     = y.size
#     resid = y - y_hat
#     s_err = np.sqrt(np.sum(resid**2) / (n-2))                    # standard deviation of the error
#     t     = stats.t.ppf(0.975, n - 2)                            # used for CI and PI bands
#     pi    = t * s_err * np.sqrt( 1 + 1/n + (x_hat - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
#     return pi


# class FrameExtractor():
#     '''
#     Class used for extracting frames from a video file.
#     '''
#     def __init__(self, video_path):
#         self.video_path = video_path
#         self.vid_cap    = cv2.VideoCapture(video_path)
#         self.n_frames   = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.fps        = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

#     def get_video_duration(self):
#         duration = self.n_frames/self.fps
#         print(f'Duration: {datetime.timedelta(seconds=duration)}')

#     def get_n_images(self, every_x_frame):
#         n_images = math.floor(self.n_frames / every_x_frame) + 1
#         print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')

#     def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext = '.jpg', frames=[100, 200]):
#         if not self.vid_cap.isOpened():
#             self.vid_cap = cv2.VideoCapture(self.video_path)

#         if dest_path is None:
#             dest_path = os.getcwd()
#         else:
#             if not os.path.isdir(dest_path):
#                 os.mkdir(dest_path); print(f'Created the following directory: {dest_path}')

#         frame_cnt = 0; img_cnt = 0
#         while self.vid_cap.isOpened():
#             success,image = self.vid_cap.read()
#             if not success: break
#             if frame_cnt % every_x_frame == 0 and (frames[0]<frame_cnt and frames[1]>frame_cnt):
#                 img_path = os.path.join(dest_path, ''.join([img_name,  '%06d' % (img_cnt+1), img_ext]))
#                 cv2.imwrite(img_path, image)
#                 img_cnt += 1
#             frame_cnt += 1
#             if(frame_cnt>frames[1]): break
#         self.vid_cap.release()
#         cv2.destroyAllWindows()


import argparse
import datetime
import math
import numpy as np
import scipy.stats as stats
import torch
import os
import cv2


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')    
        
        
        
def get_colors():    
    RGB_tuples                   = np.vstack([np.loadtxt("utils/colors.txt", skiprows=1) , np.random.uniform(0, 255, size=(10000, 3)), [[0,0,0]]])
    b                            = np.where(RGB_tuples==0)
    RGB_tuples[b]                = 1
    
    return RGB_tuples


def get_prediction_interval(y, y_hat, x, x_hat):
    n     = y.size
    resid = y - y_hat
    s_err = np.sqrt(np.sum(resid**2) / (n-2))                    # standard deviation of the error
    t     = stats.t.ppf(0.975, n - 2)                            # used for CI and PI bands
    pi    = t * s_err * np.sqrt( 1 + 1/n + (x_hat - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    return pi




class FrameExtractor():
    '''
    Class used for extracting frames from a video file.
    '''
    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap    = cv2.VideoCapture(video_path)
        self.n_frames   = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps        = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

    def get_video_duration(self):
        duration = self.n_frames/self.fps
        print(f'Duration: {datetime.timedelta(seconds=duration)}')

    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')

    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext = '.jpg', start_frame=1000, end_frame=2000):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)

        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path); print(f'Created the following directory: {dest_path}')

        frame_cnt = 0; img_cnt = 0
        while self.vid_cap.isOpened():
            success,image = self.vid_cap.read()
            if not success: break
            if frame_cnt % every_x_frame == 0 and frame_cnt > start_frame and frame_cnt < end_frame:
                img_path = os.path.join(dest_path, ''.join([img_name,  '%06d' % (img_cnt+1), img_ext]))
                cv2.imwrite(img_path, image)
                img_cnt += 1
            frame_cnt += 1
        self.vid_cap.release()
        cv2.destroyAllWindows()
        
    
def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,2,3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)
