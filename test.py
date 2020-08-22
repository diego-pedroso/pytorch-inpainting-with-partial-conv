import argparse
import torch
from torchvision import transforms

import opt
from places2 import Places2
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str)
parser.add_argument('--masks_path', type=str)
parser.add_argument('--snapshot', type=str, default='')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--width', type=int)
parser.add_argument('--height', type=int)
parser.add_argument('--cpu', type=bool, default=False)

args = parser.parse_args()

if args.cpu:
  device = torch.device('cpu')
else:
  device = torch.device('cuda')

width = args.width or args.image_size
height = args.height or args.image_size
size = (height, width)

img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

# dataset_val = Places2(args.root, args.masks_path, img_transform, mask_transform, 'val')
dataset_val = Places2(args.root, args.masks_path, img_transform, mask_transform, 'test')

model = PConvUNet().to(device)

if args.cpu:
  load_ckpt(args.snapshot, [('model', model)], device=device)
else:
  load_ckpt(args.snapshot, [('model', model)])

model.eval()
evaluate(model, dataset_val, device, 'result.jpg')
