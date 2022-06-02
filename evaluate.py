import argparse
import os
import time
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw

from utils import load_yaml, metrics, draw_detection
from datasets import YoloDataset
from datasets.voc_colors import COLORS
from models import create_model


parser = argparse.ArgumentParser(description='YOLOv1-pytorch')
parser.add_argument("--type", "-t", default="ms", help="model type", type=str)
parser.add_argument("--cfg", "-c", default="models/yolov1.yaml", help="Yolov1 config file path", type=str)
parser.add_argument("--weights", "-w", default="", help="Pretrained model weights path", type=str)
parser.add_argument("--dataset", "-d", default="datasets/voc.yaml", help="Dataset config file path", type=str)
parser.add_argument("--output", "-o", default="output", help="Output path", type=str)
parser.add_argument('--tboard', action='store_true', default=False, help='use tensorboard')
parser.add_argument("--cuda", "-cu", action='store_true', default=True, help='use cuda')
args = parser.parse_args()

torch.manual_seed(0)

def evaluate(model, val_loader, S, B, num_classes):
    model.eval()  # Sets the module in evaluation mode
    ap = 0.0
    pbar = tqdm(val_loader, leave=True)
    count = 1
    for img, labels in pbar:
        preds = model(img)[0].detach().cpu()
        ap += metrics.mean_average_precision(preds, labels, S, B, num_classes)
        pbar.set_description(f"mAP = {(ap/count):.03f}")
        count += 1
    
    tqdm.write(f"Evaluation summary -- mAP = {(ap/count):.03f}")

    
if __name__ == "__main__":
    cfg = load_yaml(args.cfg)
    print('cfg:', cfg)
    dataset = load_yaml(args.dataset)
    print('dataset:', dataset)

    img_list_path = dataset['images_eval']
    class_names = dataset['class_names']
    S, B, num_classes, input_size = cfg['S'], cfg['B'], cfg['num_classes'], cfg['input_size']

    device = 'cpu'
    if args.cuda:
        device = 'cuda:0'

    # build model
    model = create_model(args.weights, S, B, num_classes, args.type, device)

    # get data loader
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    eval_dataset = YoloDataset(img_list_path, S, B, num_classes, transform, img_box_transforms=None, eval_mode=True)
    loader = DataLoader(eval_dataset)
    evaluate(model, loader, S, B, num_classes)
