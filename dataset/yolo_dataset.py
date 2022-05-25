import os

import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from utils import encode

class YoloDataset(Dataset):
    def __init__(self, img_list_path, S, B, num_classes, transforms=None):
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
        
        self.S = S
        self.B = B
        self.num_classes = num_classes

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        # read image
        img_filename = self.img_filenames[idx]
        img = Image.open(img_filename, mode='r')
        img = self.transforms(img)  # resize and to tensor

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

        label = encode(xywhc, self.S, self.B, self.num_classes)  # convert xywhc list to label
        label = torch.Tensor(label)
        return img, label


def create_dataloader(img_list_path, train_proportion, val_proportion, test_proportion, batch_size, input_size,
                      S, B, num_classes):
    transform = transforms.Compose([
        transforms.ColorJitter(0.2, 0.4, 0.5, 0.05),
        transforms.RandomAutocontrast(0.3),
        transforms.Resize((input_size, input_size)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor()
    ])

    # create yolo dataset
    dataset = YoloDataset(img_list_path, S, B, num_classes, transforms=transform)

    dataset_size = len(dataset)
    train_size = int(dataset_size * train_proportion)
    val_size = int(dataset_size * val_proportion)
    # test_size = int(dataset_size * test_proportion)
    test_size = dataset_size - train_size - val_size

    # split dataset to train set, val set and test set three parts
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
