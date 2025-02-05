#LIA
import torch
import torch.nn as nn
from networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from utils.anonymize import anonymize
from utils.best_frame import process_video


def load_image(image, size):
    
    if isinstance(image, str):  # If input is a file path
        img = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):  # If input is already a PIL image
        img = image.convert('RGB')
    elif isinstance(image, np.ndarray):  # If input is a NumPy array
        img = Image.fromarray(image).convert('RGB')
    else:
        raise ValueError("Unsupported input type. Provide a file path, PIL Image, or NumPy array.")
    
    
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


# def vid_preprocessing(vid_path):
#     vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
#     vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
#     fps = vid_dict[2]['video_fps']
#     vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

#     return vid_norm, fps

def vid_preprocessing(vid_path, size=256, max_duration=10):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0]  # (T, H, W, C)
    fps = vid_dict[2]['video_fps']

    # Calculate the number of frames to process
    max_frames = int(fps * max_duration)
    vid = vid[:max_frames]  # Select only the first max_duration seconds

    # Resize frames to the desired size
    vid_resized = torch.nn.functional.interpolate(
        vid.permute(0, 3, 1, 2).float(),  # Convert to (T, C, H, W)
        size=(size, size),
        mode='bilinear',
        align_corners=False
    )

    # Normalize video to [-1, 1]
    vid_resized = vid_resized.unsqueeze(0)  # Add batch dimension
    vid_norm = (vid_resized / 255.0 - 0.5) * 2.0

    return vid_norm, fps



def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    torchvision.io.write_video(save_path, vid[0], fps=fps)
    


class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        self.args = args

        if args.model == 'vox':
            model_path = 'checkpoints/vox.pt'
        elif args.model == 'taichi':
            model_path = 'checkpoints/taichi.pt'
        elif args.model == 'ted':
            model_path = 'checkpoints/ted.pt'
        else:
            raise NotImplementedError
        
        print('==> searching best source image')
        source_image = process_video(args.driving_path, args.source_path)
        
        print('==> anonymizing source image')
        source_image = anonymize(args.source_path, args.anon_path)

        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        print('==> loading data')
        self.save_path = args.save_folder + '/%s' % args.model
        os.makedirs(self.save_path, exist_ok=True)
        self.save_path = os.path.join(self.save_path, Path(args.source_path).stem + '_' + Path(args.driving_path).stem + '.mp4')
        self.img_source = img_preprocessing(source_image, args.size).cuda()
        self.vid_target, self.fps = vid_preprocessing(args.driving_path)
        self.vid_target = self.vid_target.cuda()

    def run(self):

        print('==> running')
        with torch.no_grad():

            vid_target_recon = []

            if self.args.model == 'ted':
                h_start = None
            else:
                h_start = self.gen.enc.enc_motion(self.vid_target[:, 0, :, :, :])

            for i in tqdm(range(self.vid_target.size(1))):
                img_target = self.vid_target[:, i, :, :, :]
                img_recon = self.gen(self.img_source, img_target, h_start)
                vid_target_recon.append(img_recon.unsqueeze(2))

            vid_target_recon = torch.cat(vid_target_recon, dim=2)
            save_video(vid_target_recon, self.save_path, self.fps)
            


if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--source_path", type=str, default='')
    parser.add_argument("--anon_path", type=str, default='./anon')
    parser.add_argument("--driving_path", type=str, default='')
    parser.add_argument("--save_folder", type=str, default='./res')
    args = parser.parse_args()
    
    args.source_path = './source/' + Path(args.driving_path).stem + '_source' + '.png'

    # demo
    print=("LIA has started!")
    demo = Demo(args)
    #demo.run()
    print=("LIA is done!")
