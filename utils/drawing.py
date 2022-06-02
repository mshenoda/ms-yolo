import imp
import cv2
import numpy as np
from PIL import Image
from datasets.voc_colors import COLORS

def draw_detection(img, bboxes, class_names, colors=None):
    if colors is None:
        colors = COLORS
    if img is Image:
        img = np.array(img)
    h, w = img.shape[0:2]
    n = bboxes.size()[0]
    bboxes = bboxes.detach().numpy()
    print(bboxes)
    for i in range(n):
        p1 = (int((bboxes[i, 0] - bboxes[i, 2] / 2) * w), int((bboxes[i, 1] - bboxes[i, 3] / 2) * h))
        p2 = (int((bboxes[i, 0] + bboxes[i, 2] / 2) * w), int((bboxes[i, 1] + bboxes[i, 3] / 2) * h))
        class_name = class_names[int(bboxes[i, 5])]
        cv2.rectangle(img, p1, p2, color=colors[int(bboxes[i, 5])], thickness=2)
        cv2.putText(img, class_name, p1, cv2.FONT_HERSHEY_TRIPLEX, 0.9, colors[int(bboxes[i, 5])])
    return img