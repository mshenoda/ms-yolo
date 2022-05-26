import argparse
import os
import time
import cv2
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from utils import load_yaml
from torchvision import transforms
from utils.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomBlur

parser = argparse.ArgumentParser(description='YOLOv1-pytorch')
parser.add_argument("--cfg", "-c", default="models/yolov1.yaml", help="Yolov1 config file path", type=str)
parser.add_argument("--dataset", "-d", default="datasets/voc.yaml", help="Dataset config file path", type=str)
parser.add_argument("--output", "-o", default="output", help="Output path", type=str)
parser.add_argument("--batch_size", "-bs", default=4, help="Training batch size", type=int)

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
        # if self.transforms is not None:
        #     img = self.transforms(img)

        # read each image's corresponding label(.txt)
        xywhc = []
        with open(self.label_files[idx], 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    continue
                line = line.strip().split(' ')

                # convert xywh str to float, class str to int
                c, x, y, w, h = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])

                xywhc.append((x, y, w, h, c))
                
        if self.img_box_transforms is not None:
            for t in self.img_box_transforms:
                img, xywhc = t(img, xywhc)
        label = torch.Tensor(xywhc)
        return img, label


def test(idx, img_filenames, label_files):
    img_filename = img_filenames[idx]
    img = cv2.imread(img_filename, mode='r')
    xywhc = []
    with open(label_files[idx], 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            line = line.strip().split(' ')

            # convert xywh str to float, class str to int
            c, x, y, w, h = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])

            xywhc.append((x, y, w, h, c))
    

if __name__ == "__main__":
    cfg = load_yaml(args.cfg)
    print('cfg:', cfg)
    dataset = load_yaml(args.dataset)
    print('dataset:', dataset)
    #img_path, label_path = dataset_cfg['images'], dataset_cfg['labels']
    img_list_path = dataset['images']
    S, B, num_classes, input_size = cfg['S'], cfg['B'], cfg['num_classes'], cfg['input_size']

    # create output file folder
    start = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    output_path = os.path.join(args.output, 'train', start)
    os.makedirs(output_path)
    
    # transform = transforms.Compose([
    #     transforms.ColorJitter(0.2, 0.7, 0.7, 0.1),
    #     transforms.RandomAutocontrast(0.3),
    #     RandomBlur(kernel_size=[5,5], sigma=[0.2, 2], p=0.2),
    #     transforms.RandomGrayscale(p=0.1),
    #     transforms.Resize((input_size, input_size)),
    #     transforms.ToTensor()
    # ])
    # img_box_transform = [
    #     RandomHorizontalFlip(0.5),
    #     RandomVerticalFlip(0.1)
    # ]

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

    for idx in len(label_files):
        test(idx, img_filenames, label_files)
