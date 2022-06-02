import argparse
import os
import shutil
import torch
import cv2
from utils import load_yaml, draw_detection
from utils.detector import Detector
from datasets.voc_colors import COLORS

parser = argparse.ArgumentParser(description='YOLO MS')

parser.add_argument("--type", "-t", default="ms", help="model type", type=str)
parser.add_argument("--weights", "-w", default="weights/yolo_ms.pt", help="Path of model weight", type=str)
parser.add_argument("--source", "-s", default="data/samples", help="Path of your input image, video, directory", type=str)
parser.add_argument('--output', "-o", default='output', help='Output folder', type=str)
parser.add_argument("--cfg", "-c", default="models/yolov1.yaml", help="Your model config path", type=str)
parser.add_argument("--dataset", "-d", default="datasets/voc.yaml", help="Your dataset config path", type=str)
parser.add_argument('--conf', "-cnf", default=0.10, help='prediction confidence thresh', type=float)
parser.add_argument('--iou', "-iou", default=0.3, help='prediction iou thresh', type=float)
parser.add_argument("--cuda", "-cu", action='store_true', default=True, help='use cuda')

args = parser.parse_args()

torch.manual_seed(33)


if __name__ == "__main__":
    # load configs from config file
    cfg = load_yaml(args.cfg)
    print('cfg:', cfg)
    input_size = cfg['input_size']
    dataset = load_yaml(args.dataset)
    print('--------------------------------------------------------------------------------')
    print('dataset:', dataset)
    print('--------------------------------------------------------------------------------')
    class_names = dataset['class_names']
    S, B, num_classes = cfg['S'], cfg['B'], cfg['num_classes']
    conf, iou, source = args.conf, args.iou, args.source

    device = 'cpu'
    if args.cuda:
        device = 'cuda:0'

    # create output folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    detector = Detector(args.weights, input_size, S, B, num_classes, args.type, device)
    print('Detector created successfully using model weights: [', args.weights,']')
    print('================================================================================')
    
    # Image Detection
    if source.split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif']:
        img = cv2.imread(source)
        img_name = os.path.basename(source)

        detection = detector.detect(img, conf, iou)
        if detection.size()[0] != 0:
            img = draw_detection(img, detection, class_names, COLORS)
            # save output img
            cv2.imwrite(os.path.join(args.output, img_name), img)

    # Image Directory Detection
    elif source == source.split('.')[-1]:
        # create output folder
        output = os.path.join(args.output, source.split('/')[-1])
        if os.path.exists(output):
            shutil.rmtree(output)
            # os.removedirs(output)
        os.makedirs(output)

        imgs = os.listdir(source)
        for img_name in imgs:
            img = cv2.imread(os.path.join(source, img_name))
            detection = detector.detect(img, conf, iou)
            if detection.size()[0] != 0:
                img = draw_detection(img.copy(), detection, class_names, COLORS)
                # save output img
                cv2.imwrite(os.path.join(output, img_name), img)
            print(img_name)
