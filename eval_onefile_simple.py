import os
import cv2
import time
import joblib
import argparse
import warnings
import traceback
import numpy as np

from PHALP import PHALP_tracker
from deep_sort_ import nn_matching
from deep_sort_.detection import Detection
from deep_sort_.tracker import Tracker

from utils.utils import FrameExtractor, str2bool

warnings.filterwarnings('ignore')
  
def debug_frame(final_visuals_dic, track_id_frames):
    def get_id_color(index):
        temp_index = abs(int(index)) * 3
        color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
                (29 * temp_index) % 255)
        return color

    t_           = final_visuals_dic['time']
    debug_image  = final_visuals_dic['frame'].copy()
    tracked_ids  = final_visuals_dic["tid"]
    tracked_bbox = final_visuals_dic["bbox"]
    tracked_conf = final_visuals_dic["conf"]
    tracked_time = final_visuals_dic["tracked_time"]

    img_height, img_width, _      = debug_image.shape
    frame_size                    = (img_width, img_height)    

    for tracker_id, bbox, score, track_time in zip(tracked_ids, tracked_bbox, tracked_conf, tracked_time):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
        
        color = get_id_color(tracker_id)

        # Bounding Box
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
        )
        # トラックID、スコア
        score_txt = str(round(score, 2))
        text = f'Track ID:{tracker_id}({score_txt})'
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness=2,
        )
        # トラッキング時間
        text = f'Time:{track_time}'
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness=2,
        )

        # ID別のフレームを記録
        crop = debug_image[max(y1-30, 0):y2, x1:x2, :]
        if not tracker_id in track_id_frames.keys():
            track_id_frames[tracker_id] = [crop]
        else:
            track_id_frames[tracker_id].append(crop)

    # 推論時間
    text = f'Current time: {t_}'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image, frame_size

def write_trackid_wise_video(
    track_id_frames_dict,
    base_dir, cap_fps
):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for track_id, track_frames in track_id_frames_dict.items():
        width, height = 0, 0
        for frame in track_frames:
            height = max(frame.shape[0], height)
            width = max(frame.shape[1], width)

        writer = cv2.VideoWriter(
            f"{base_dir}/{track_id}.webm",
            cv2.VideoWriter_fourcc(*'vp80'),
            cap_fps, (width, height)
        )
        for frame in track_frames:
            pad_left = (width - frame.shape[1])//2
            pad_right = width - frame.shape[1] - pad_left
            pad_top = (height - frame.shape[0])//2
            pad_bottom = height - frame.shape[0] - pad_top
            frame_pad = cv2.copyMakeBorder(frame, 
                            pad_top, pad_bottom, pad_left, pad_right, 
                            cv2.BORDER_CONSTANT, (0, 0, 0))
            writer.write(frame_pad)
        writer.release()

def test_tracker(opt, phalp_tracker: PHALP_tracker):
    
    visual_store_   = ['tracked_ids', 'tracked_bbox', 'tid', 'bbox', 'tracked_time', 'conf']    
                                                
    if(not(opt.overwrite) and os.path.isfile('out/' + opt.storage_folder + '/results/' + str(opt.video_seq) + '.pkl')): return 0
    print(opt.storage_folder + '/results/' + str(opt.video_seq))
    
    try:
        os.makedirs('out/' + opt.storage_folder, exist_ok=True)  
        os.makedirs('out/' + opt.storage_folder + '/results', exist_ok=True)  
        os.makedirs('out/' + opt.storage_folder + '/_TMP', exist_ok=True)  
    except: pass
    
    phalp_tracker.eval()
    phalp_tracker.HMAR.reset_nmr(opt.res)    
    
    metric  = nn_matching.NearestNeighborDistanceMetric(opt, opt.hungarian_th, opt.past_lookback)
    tracker = Tracker(opt, metric, max_age=opt.max_age_track, n_init=opt.n_init, phalp_tracker=phalp_tracker, dims=[4096, 4096, 99])  
        
    try: 
        
        main_path_to_frames = opt.base_path + '/' + opt.video_seq + opt.sample
        list_of_frames      = np.sort([i for i in os.listdir(main_path_to_frames) if '.jpg' in i])
        list_of_frames      = list_of_frames if opt.start_frame==-1 else list_of_frames[opt.start_frame:opt.end_frame]
            
        tracked_frames          = []
        final_visuals_dic       = {}
        track_id_frame = {}

        for t_, frame_name in enumerate(list_of_frames):
            if(opt.verbose): 
                print('\n\n\nTime: ', opt.video_seq, frame_name, t_, time.time()-time_ if t_>0 else 0 )
                time_ = time.time()
            
            image_frame               = cv2.imread(main_path_to_frames + '/' + frame_name)
            img_height, img_width, _  = image_frame.shape
            new_image_size            = max(img_height, img_width)
            top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
            measurments               = [img_height, img_width, new_image_size, left, top]

            ############ detection ##############
            pred_bbox, pred_masks, pred_scores, mask_names, gt = phalp_tracker.get_detections(image_frame, frame_name, t_)
            
            ############ HMAR ##############
            detections = []
            for bbox, mask, score, mask_name, gt_id in zip(pred_bbox, pred_masks, pred_scores, mask_names, gt):
                if bbox[2]-bbox[0]<50 or bbox[3]-bbox[1]<100: continue
                detection_data = phalp_tracker.get_human_apl(image_frame, mask, bbox, score, [main_path_to_frames, frame_name], mask_name, t_, measurments, gt_id)
                detections.append(Detection(detection_data))

            ############ tracking ##############
            tracker.predict()
            tracker.update(detections, t_, frame_name, 0)

            ############ record the results ##############
            final_visuals_dic.setdefault(frame_name, {'time': t_})
            if(opt.render): final_visuals_dic[frame_name]['frame'] = image_frame
            for key_ in visual_store_: final_visuals_dic[frame_name][key_] = []
            
            for tracks_ in tracker.tracks:
                if(frame_name not in tracked_frames): tracked_frames.append(frame_name)
                if(not(tracks_.is_confirmed())): continue
                
                track_id        = tracks_.track_id
                track_data_hist = tracks_.track_data['history'][-1]

                final_visuals_dic[frame_name]['tid'].append(track_id)
                final_visuals_dic[frame_name]['bbox'].append(track_data_hist['bbox'])
                final_visuals_dic[frame_name]['conf'].append(track_data_hist['conf'])
                final_visuals_dic[frame_name]['tracked_time'].append(tracks_.time_since_update)

                if(tracks_.time_since_update==0):
                    final_visuals_dic[frame_name]['tracked_ids'].append(track_id)
                    final_visuals_dic[frame_name]['tracked_bbox'].append(track_data_hist['bbox'])
                    
                    if(tracks_.hits==opt.n_init):
                        for pt in range(opt.n_init-1):
                            track_data_hist_ = tracks_.track_data['history'][-2-pt]
                            frame_name_      = tracked_frames[-2-pt]
                            final_visuals_dic[frame_name_]['tid'].append(track_id)
                            final_visuals_dic[frame_name_]['bbox'].append(track_data_hist_['bbox'])
                            final_visuals_dic[frame_name_]['conf'].append(track_data_hist_['conf'])
                            final_visuals_dic[frame_name_]['tracked_ids'].append(track_id)
                            final_visuals_dic[frame_name_]['tracked_bbox'].append(track_data_hist_['bbox'])
                            final_visuals_dic[frame_name_]['tracked_time'].append(0)
                            
                            
            ############ save the video ##############
            if(opt.render and t_>=opt.n_init):
                d_ = opt.n_init+1 if(t_+1==len(list_of_frames)) else 1
                for t__ in range(t_, t_+d_):
                    frame_key          = list_of_frames[t__-opt.n_init]
                    rendered_, f_size  = debug_frame(final_visuals_dic[frame_key], track_id_frame)      
                    if(t__-opt.n_init==0):
                        file_name      = 'out/' + opt.storage_folder + '/PHALP_' + str(opt.video_seq) + '_'+ str(opt.detection_type) + '.webm'
                        os.makedirs(os.path.dirname(file_name), exist_ok=True)    
                        video_file     = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'vp80'), opt.render_fps, frameSize=f_size)
                    video_file.write(rendered_)
                    del final_visuals_dic[frame_key]['frame']

        joblib.dump(final_visuals_dic, 'out/' + opt.storage_folder + '/results/' + opt.track_dataset + "_" + str(opt.video_seq) + opt.post_fix  + '.pkl')
        if(opt.use_gt): joblib.dump(tracker.tracked_cost, 'out/' + opt.storage_folder + '/results/' + str(opt.video_seq) + '_' + str(opt.start_frame) + '_distance.pkl')
        if(opt.render): video_file.release()

        write_trackid_wise_video(
            track_id_frame, 
            f"out/{opt.storage_folder}/{opt.video_seq}", 
            opt.render_fps)

        
    except Exception as e: 
        print(e)
        print(traceback.format_exc())     
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PHALP_pixel Tracker')
    parser.add_argument('--batch_id', type=int, default='-1')
    parser.add_argument('--track_dataset', type=str, default='posetrack')
    parser.add_argument('--predict', type=str, default='APL')
    parser.add_argument('--storage_folder', type=str, default='Videos_v20.000')
    parser.add_argument('--distance_type', type=str, default='A5')
    parser.add_argument('--use_gt', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--overwrite', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--low_th_c', type=float, default=0.95)
    parser.add_argument('--hungarian_th', type=float, default=100.0)
    parser.add_argument('--track_history', type=int, default=7)
    parser.add_argument('--max_age_track', type=int, default=20)
    parser.add_argument('--n_init',  type=int, default=1)
    parser.add_argument('--max_ids', type=int, default=50)
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False)
    
    parser.add_argument('--base_path', type=str)
    parser.add_argument('--video_seq', type=str, default='_DATA/posetrack/list_videos_val.npy')
    parser.add_argument('--youtube_id', type=str, default="xEH_5T9jMVU")
    parser.add_argument('--all_videos', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--store_mask', type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("--video_path", type=str, default="")

    parser.add_argument('--render', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--render_type', type=str, default='HUMAN_HEAD_FAST')
    parser.add_argument('--render_up_scale', type=int, default=2)
    parser.add_argument('--res', type=int, default=256)
    parser.add_argument('--downsample',  type=int, default=1)
    parser.add_argument("--render_fps", type=int, default=30)
    
    parser.add_argument('--encode_type', type=str, default='3c')
    parser.add_argument('--cva_type', type=str, default='least_square')
    parser.add_argument('--past_lookback', type=int, default=1)
    parser.add_argument('--mask_type', type=str, default='feat')
    parser.add_argument('--detection_type', type=str, default='mask2')
    parser.add_argument('--start_frame', type=int, default='-1')
    parser.add_argument('--end_frame', type=int, default='-1')
    parser.add_argument('--store_extra_info', type=str2bool, nargs='?', const=True, default=False)
    
    opt                   = parser.parse_args()
    opt.sample            = ''
    opt.post_fix          = ''
    
    phalp_tracker         = PHALP_tracker(opt)
    phalp_tracker.cuda()
    phalp_tracker.eval()

    video = os.path.splitext(os.path.basename(opt.video_path))[0]
    os.system("rm -rf " + "_DEMO/" + video)
    os.makedirs("_DEMO/" + video, exist_ok=True)    
    os.makedirs("_DEMO/" + video + "/img", exist_ok=True)    
    fe = FrameExtractor(opt.video_path)
    print('Number of frames: ', fe.n_frames)
    start_frame = opt.start_frame if opt.start_frame > 0 else 0
    end_frame = opt.end_frame if opt.end_frame > 0 else fe.n_frames
    fe.extract_frames(every_x_frame=1, img_name='', dest_path= "_DEMO/" + video + "/img/", start_frame=start_frame, end_frame=end_frame)

    opt.base_path       = '_DEMO/'
    opt.video_seq       = video
    opt.sample          =  '/img/'
    test_tracker(opt, phalp_tracker)