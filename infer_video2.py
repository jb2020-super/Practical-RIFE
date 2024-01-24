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

warnings.filterwarnings("ignore")

def pad_image(img, padding, fp16):
    if fp16:
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)


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
    with tqdm(total=int(total_frame)) as pbar:
        while not eof:
            rst, second = vc.read()
            if not rst:
                eof = True
                break
            first_tensor = torch.from_numpy(np.transpose(first, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            first_tensor = pad_image(first_tensor, padding, False)
            second_tensor = torch.from_numpy(np.transpose(second, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            second_tensor = pad_image(second_tensor, padding, False)

            mid = model.inference(first_tensor, second_tensor, 0.5, 1.0)
            mid = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
            mid = mid[:height, :, :]
            vw.write(first)
            vw.write(mid)
            first = second
            pbar.update(1)




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