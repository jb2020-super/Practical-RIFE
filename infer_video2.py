import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
from train_log.RIFE_HDv3 import Model
from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector
from scenedetect import open_video, SceneManager

warnings.filterwarnings("ignore")

def pad_image(img, padding, fp16):
    if fp16:
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

def infer_once(model, first, second, exp):
    mid = model.inference(first, second, 0.5, 1.0)
    exp -= 1
    rst = []
    if exp >= 1:
        first_list = infer_once(model, first, mid, exp)
        second_list = infer_once(model, mid, second, exp)
        rst += first_list
        rst.append(mid)
        rst += second_list
    else:
        rst.append(mid)
    return rst

def scene_detect(first, second):
    # Convert frames to grayscale
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between frames
    diff = cv2.absdiff(first_gray, second_gray)

    # Set a threshold to identify significant changes
    _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Count non-zero pixels in the thresholded difference
    non_zero_count = cv2.countNonZero(threshold_diff)
    print(non_zero_count)
scene_list = []
cnt = 1
# Callback to invoke on the first frame of every new scene detection.
def on_new_scene(frame_img: np.ndarray, frame_num: int):
    global cnt, scene_list
    print("{} New scene found at frame {}.".format(cnt, frame_num))
    cnt += 1
    scene_list.append(frame_num)
    
def get_scene_list(video_path):
    

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=1.5, min_scene_len=5))
    scene_manager.detect_scenes(video=video, callback=on_new_scene)
    return scene_list

def infer_video(model, input, output, exp, is_slomo):
    vc = cv2.VideoCapture(input)
    if not vc.isOpened():
        print('Open file {} failed!'.format(input))
        exit()
    fps = vc.get(cv2.CAP_PROP_FPS)
    total_frame = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_width = width
    out_height = height
    out_fps = fps
    if not is_slomo:
        out_fps = fps * (2 ** exp)
    vw = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (out_width, out_height))

    tmp = max(128, int(128 / args.scale))
    ph = ((height - 1) // tmp + 1) * tmp
    pw = ((width - 1) // tmp + 1) * tmp
    padding = (0, pw - width, 0, ph - height)

    rst, first = vc.read()
    eof = not rst
    ada_detector = AdaptiveDetector(adaptive_threshold=1.5, min_scene_len=5)
    scene_list = get_scene_list(input)
    fno = 0
    scene_idx = 0
    with tqdm(total=int(total_frame)) as pbar:
        while not eof:
            rst, second = vc.read()
            if not rst:
                eof = True
                break
            fno += 1
            is_scene_change = False
            if scene_idx < len(scene_list) and fno == scene_list[scene_idx]:
                is_scene_change = True
                scene_idx += 1
            #is_scene_change = scene_detect(first, second)

            if is_scene_change:
                n = 2 ** exp
                step = 1.0 / n
                for alpha in np.arange(step, 1, step):
                    mid = cv2.addWeighted(first, 1 - alpha, second, alpha, 0)
                    vw.write(mid)
            else:
                first_tensor = torch.from_numpy(np.transpose(first, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
                first_tensor = pad_image(first_tensor, padding, False)
                second_tensor = torch.from_numpy(np.transpose(second, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
                second_tensor = pad_image(second_tensor, padding, False)

                mid_list = infer_once(model, first_tensor, second_tensor, args.exp)
                vw.write(first)
                for mid in mid_list:
                    mid = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
                    mid = mid[:height, :, :]
                    vw.write(mid)
                    
            first = second
            pbar.update(1)


# python .\infer_video2.py -i d:\VFI-Data\VFITestDataSet\vimeo\AUTOPILOTS-25fps.mp4 -o D:\VFI-Data\RIFEv4.14\VFITestDataSet\autopilots-4x-slomo-v4-blend.mp4 --slomo --exp 2
#

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('-i', '--input', type=str, help="input video file path, or folder")
    argp.add_argument('-o', '--output', type=str, default='output.mp4', help="output file path, or folder")
    argp.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
    argp.add_argument('--exp', type=int, default=1, help='The interpolation factor 2^exp')
    #argp.add_argument('--fps', dest='fps', type=int, default=None)
    argp.add_argument('--slomo', action='store_true', help='output as slow motion')
    argp.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
    args = argp.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model()
    if not hasattr(model, 'version'):
        model.version = 0
    model.load_model(args.modelDir, -1)
    print("Loaded 3.x/4.x HD model.")
    model.eval()
    model.device()
    with torch.no_grad():
        infer_video(model, args.input, args.output, args.exp, args.slomo)