import argparse
import imp
import os
import random
import shutil
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from utils import load_yaml, decode, draw_bbox
from models import create_model

parser = argparse.ArgumentParser(description='YOLOv1 Pytorch Implementation')
parser.add_argument("--weights", "-w", default="weights/old/epoch17.pt", help="Path of model weight", type=str)
parser.add_argument("--source", "-s", default="data/samples", help="Path of your input image, video, directory", type=str)
parser.add_argument('--output', "-o", default='output', help='Output folder', type=str)
parser.add_argument("--cfg", "-c", default="models/yolov1.yaml", help="Your model config path", type=str)
parser.add_argument("--dataset", "-d", default="datasets/voc.yaml", help="Your dataset config path", type=str)
parser.add_argument('--conf_thresh', "-ct", default=0.25, help='prediction confidence thresh', type=float)
parser.add_argument('--iou_thresh', "-it", default=0.45, help='prediction iou thresh', type=float)
args = parser.parse_args()

torch.manual_seed(32)

# random colors
COLORS = [
[0, 0, 255],
[255, 0, 0],
[255, 140, 0],
[221, 160, 221],
[186, 85, 211],
[0, 250, 154],
[0, 255, 255],
[255, 0, 255],
[30, 144, 255],
[250, 128, 114],
[255, 255, 84],
[127, 255, 0],
[255, 20, 147],
[135, 206, 250],
[250, 25, 215],
[0, 100, 0],
[47, 79, 79],
[46, 139, 87],
[127, 0, 0],
[128, 128, 0],
[0, 0, 139],
]

def detect(img, model, input_size, S, B, num_classes, conf_thresh, iou_thresh):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred_img = Image.fromarray(img).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
    pred_img = transform(pred_img)
    pred_img.unsqueeze_(0)

    pred = model(pred_img)[0].detach().cpu()
    xywhcc = decode(pred, S, B, num_classes, conf_thresh, iou_thresh)

    return xywhcc


if __name__ == "__main__":
    # load configs from config file
    cfg = load_yaml(args.cfg)
    print('cfg:', cfg)
    input_size = cfg['input_size']
    dataset = load_yaml(args.dataset)
    print('dataset:', dataset)
    class_names = dataset['class_names']
    print('Class names:', class_names)
    S, B, num_classes = cfg['S'], cfg['B'], cfg['num_classes']
    conf_thresh, iou_thresh, source = args.conf_thresh, args.iou_thresh, args.source

    # load model
    model = create_model(args.weights, S, B, num_classes)
    print('Model loaded successfully!')

    # create output folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Image
    if source.split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp']:
        img = cv2.imread(source)
        img_name = os.path.basename(source)

        xywhcc = detect(img, model, input_size, S, B, num_classes, conf_thresh, iou_thresh)
        if xywhcc.size()[0] != 0:
            img = draw_bbox(img, xywhcc, class_names, COLORS)
            # save output img
            cv2.imwrite(os.path.join(args.output, img_name), img)

    # Video
    elif source.split('.')[-1] in ['avi', 'mp4', 'mkv', 'flv', 'mov']:
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('Video loaded failed!')
                break

            xywhcc = detect(frame, model, input_size, S, B, num_classes, conf_thresh, iou_thresh)
            if xywhcc.size()[0] != 0:
                frame = draw_bbox(frame, xywhcc, class_names, COLORS)

            cv2.resizeWindow('frame', int(cap.get(3)), int(cap.get(4)))
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # Folder
    elif source == source.split('.')[-1]:
        # create output folder
        output = os.path.join(args.output, source.split('/')[-1])
        if os.path.exists(output):
            shutil.rmtree(output)
            # os.removedirs(output)
        os.makedirs(output)

        imgs = os.listdir(source)
        for img_name in imgs:
            # img = cv2.imread(os.path.join(source, img_name))
            img = cv2.imdecode(np.fromfile(os.path.join(
                source, img_name), dtype=np.uint8), cv2.IMREAD_COLOR)
            # predict
            xywhcc = detect(img, model, input_size, S, B, num_classes, conf_thresh, iou_thresh)
            if xywhcc.size()[0] != 0:
                img = draw_bbox(img.copy(), xywhcc, class_names, COLORS)
                # save output img
                cv2.imwrite(os.path.join(output, img_name), img)
            print(img_name)
