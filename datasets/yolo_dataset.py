import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import encode

class YoloDataset(Dataset):
    def __init__(self, img_list_path, S, B, num_classes, transforms=None, img_box_transforms=None, eval_mode=False):
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
        self.eval_mode = eval_mode

    def eval(self, eval_mode=True):
        self.eval_mode = eval_mode
        return

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        # read image
        img_filename = self.img_filenames[idx]
        img = Image.open(img_filename, mode='r')
        if self.transforms is not None:
            img = self.transforms(img)

        # read each image's corresponding label (.txt)
        labels = []
        with open(self.label_files[idx], 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    continue
                line = line.strip().split(' ')
                # convert xywh str to float, class str to int
                c, x, y, w, h = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
                if self.eval_mode: 
                    labels.append((x, y, w, h, 1.0, c))
                else:
                    labels.append((x, y, w, h, c))
                
        if self.img_box_transforms is not None:
            for t in self.img_box_transforms:
                img, labels = t(img, labels)
        
        if self.eval_mode: 
            return img, torch.Tensor(labels)

        encoded_labels = encode(labels, self.S, self.B, self.num_classes)  # convert label list to encoded label
        encoded_labels = torch.Tensor(encoded_labels)
        return img, encoded_labels
