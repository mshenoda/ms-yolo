import argparse
import os
import time
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils import load_yaml
from torchvision import transforms
from utils.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomBlur, RandomRotationJitter
from datasets.voc_colors import COLORS

parser = argparse.ArgumentParser(description='YOLO')
parser.add_argument("--cfg", "-c", default="config/model.yaml", help="model config file", type=str)
parser.add_argument("--dataset", "-d", default="config/dataset.yaml", help="dataset config file", type=str)
parser.add_argument("--batch_size", "-bs", default=4, help="batch size", type=int)

args = parser.parse_args()

torch.manual_seed(32)

class YoloTestDataset(Dataset):
    def __init__(self, img_list_path, S, B, num_classes, transforms=None, img_box_transforms=None):
        with open(img_list_path, "r") as img_list_file:
            self.img_filenames = img_list_file.readlines()
            
        self.img_filenames = list(map(lambda x:x.strip(), self.img_filenames))
        self.label_files = []
        for path in self.img_filenames:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.transforms = transforms
        self.img_box_transforms = img_box_transforms
        
        self.S = S
        self.B = B
        self.num_classes = num_classes

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        # read image
        img_filename = self.img_filenames[idx]
        img = Image.open(img_filename, mode='r')
  
        bbox = []
        with open(self.label_files[idx], 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    continue
                line = line.strip().split(' ')
                c, x, y, w, h = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
                bbox.append((x, y, w, h, c))
                
        if self.img_box_transforms is not None:
            for t in self.img_box_transforms:
                img, bbox = t(img, bbox)
        tensor = torch.Tensor(bbox)
        return img, tensor

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

def get_img_labels(idx, img_filenames, label_files):
    img_filename = img_filenames[idx]
    #img = cv2.imread(img_filename)
    img = Image.open(img_filename, mode='r')
    labels = []
    with open(label_files[idx], 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            line = line.strip().split(' ')
            c, x, y, w, h = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
            labels.append((x, y, w, h, c))
    return img, labels


if __name__ == "__main__":
    cfg = load_yaml(args.cfg)
    print('cfg:', cfg)
    dataset = load_yaml(args.dataset)
    print('dataset:', dataset)

    img_list_path = dataset['images']
    S, B, num_classes, input_size = cfg['S'], cfg['B'], cfg['num_classes'], cfg['input_size']

    transform = [
        transforms.Resize((input_size, input_size)),
        transforms.ColorJitter(0.2, 0.7, 0.7, 0.1),
        transforms.RandomAdjustSharpness(3, p=0.2),
        RandomBlur(kernel_size=[3,3], sigma=[0.1, 2], p=0.1),
        transforms.RandomGrayscale(p=0.1)
    ]
    img_box_transform = [
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.05),
        RandomRotationJitter()
    ]

    with open(img_list_path, "r") as img_list_file:
        img_filenames = img_list_file.readlines()

    img_filenames = list(map(lambda x:x.strip(), img_filenames))
    label_files = []
    for path in img_filenames:
        image_dir = os.path.dirname(path)
        label_dir = "labels".join(image_dir.rsplit("images", 1))
        assert label_dir != image_dir, \
            f"Image path must contain a folder named 'images'! \n'{image_dir}'"
        label_file = os.path.join(label_dir, os.path.basename(path))
        label_file = os.path.splitext(label_file)[0] + '.txt'
        label_files.append(label_file)

    idx = 0
    while idx < len(label_files):
        img, labels = get_img_labels(idx, img_filenames, label_files)
        for t in transform:
            img = t(img)
        for ibt in img_box_transform:
            img, labels = ibt(img, labels)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for x, y, w, h, c in labels:
            p1, p2 = absolute_points(x, y, w, h, img_size=img.shape)
            img = cv2.rectangle(img, p1, p2, color=COLORS[c], thickness=2)
            img = cv2.putText(img, str(c), p1, cv2.FONT_HERSHEY_TRIPLEX, 0.9, COLORS[int(c)])
        cv2.imshow("img", img)
        key = cv2.waitKey()
        if key == ord('a'): # next
            idx -=1 
            continue
        if key == ord('d'): # previous
            idx +=1 
            continue
        if key == ord('q'): # quit
            break
        idx += 1
