import argparse
import os
import shutil
import torch
import cv2
from utils import load_yaml, draw_detection
from utils.detector import Detector

parser = argparse.ArgumentParser(description='YOLOv1 Pytorch Implementation')
parser.add_argument("--weights", "-w", default="weights/epoch17.pt", help="Path of model weight", type=str)
parser.add_argument("--source", "-s", default="data/samples", help="Path of your input image, video, directory", type=str)
parser.add_argument('--output', "-o", default='output', help='Output folder', type=str)
parser.add_argument("--cfg", "-c", default="models/yolov1.yaml", help="Your model config path", type=str)
parser.add_argument("--dataset", "-d", default="datasets/voc.yaml", help="Your dataset config path", type=str)
parser.add_argument('--conf', "-cnf", default=0.125, help='prediction confidence thresh', type=float)
parser.add_argument('--iou', "-iou", default=0.35, help='prediction iou thresh', type=float)
args = parser.parse_args()

torch.manual_seed(33)

# 20 colors
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

    # create output folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    detector = Detector(args.weights, input_size, S, B, num_classes)
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

    # Video Detection
    elif source.split('.')[-1] in ['avi', 'mp4', 'mkv', 'mov', 'wmv', 'flv']:
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('Video loaded failed!')
                break
            
            frame = cv2.resize(frame, [640,480])
            detection = detector.detect(frame, conf, iou)
            if detection.size()[0] != 0:
                frame = draw_detection(frame, detection, class_names, COLORS)

            #cv2.resizeWindow('frame', int(cap.get(3)), int(cap.get(4)))
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

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
            # img = cv2.imdecode(np.fromfile(os.path.join(
            #     source, img_name), dtype=np.uint8), cv2.IMREAD_COLOR)
            # predict
            detection = detector.detect(img, conf, iou)
            if detection.size()[0] != 0:
                img = draw_detection(img.copy(), detection, class_names, COLORS)
                # save output img
                cv2.imwrite(os.path.join(output, img_name), img)
            print(img_name)
