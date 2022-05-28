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
parser.add_argument("--cfg", "-c", default="models/yolov1.yaml", help="Yolov1 config file path", type=str)
parser.add_argument("--weights", "-w", default="./weights/new/epoch78.pt", help="Pretrained model weights path", type=str)
parser.add_argument("--dataset", "-d", default="datasets/voc.yaml", help="Dataset config file path", type=str)
parser.add_argument("--output", "-o", default="output", help="Output path", type=str)
parser.add_argument("--epochs", "-e", default=135, help="Training epochs", type=int)
parser.add_argument("--lr", "-lr", default=0.0005, help="Training learning rate", type=float)
parser.add_argument("--batch_size", "-bs", default=64, help="Training batch size", type=int)
parser.add_argument("--checkpoint", "-cp", default=1, help="Frequency of saving model checkpoint when training", type=int)
parser.add_argument('--tboard', action='store_true', default=False, help='use tensorboard')
parser.add_argument("--cuda", "-cu", action='store_true', default=True, help='use cuda')
args = parser.parse_args()

torch.manual_seed(32)

def absolute_points(x, y, w, h, img_size):
    imgW = img_size[1]
    imgH = img_size[0]
    x1, x2 = x - (w / 2), x + (w / 2)
    y1, y2 = y - (h / 2), y + (h / 2)
    x1 *= imgW
    x2 *= imgW
    y1 *= imgH
    y2 *= imgH
    return (int(x1), int(y1)), (int(x2), int(y2))

def evaluate(model, val_loader, S, B, class_names):
    model.eval()  # Sets the module in evaluation mode
    ap = 0.0
    pbar = tqdm(val_loader, leave=True)
    count = 1
    for img, labels in pbar:
        #img = img.to(device)#, targets#.to(device)
        preds = model(img)[0].detach().cpu()
        ap += metrics.average_precision(preds, labels)
        pbar.set_description(f"mAP = {(ap/count):.03f}")
        count += 1
    
    tqdm.write(f"Evaluation summary -- mAP = {(ap/count):.03f}")

    

if __name__ == "__main__":
    cfg = load_yaml(args.cfg)
    print('cfg:', cfg)
    dataset = load_yaml(args.dataset)
    print('dataset:', dataset)
    #img_path, label_path = dataset_cfg['images'], dataset_cfg['labels']
    img_list_path = dataset['images_eval']
    class_names = dataset['class_names']
    S, B, num_classes, input_size = cfg['S'], cfg['B'], cfg['num_classes'], cfg['input_size']

    # create output file folder
    start = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    output_path = os.path.join(args.output, 'train', start)
    os.makedirs(output_path)
    
    device = torch.device("cpu")
    
    # build model
    model = create_model(args.weights, S, B, num_classes).to(device)

    # get data loader
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    eval_dataset = YoloDataset(img_list_path, S, B, num_classes, transform, img_box_transforms=None, eval_mode=True)
    loader = DataLoader(eval_dataset)
    evaluate(model, loader, class_names, device)
