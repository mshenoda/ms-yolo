import imp
import cv2
import numpy as np
from PIL import Image
from datasets.voc_colors import COLORS

def draw_detection(img, bboxs, class_names, colors=None):
    if colors is None:
        colors = COLORS
    if img is Image:
        img = np.array(img)
    h, w = img.shape[0:2]
    n = bboxs.size()[0]
    bboxs = bboxs.detach().numpy()
    print(bboxs)
    for i in range(n):
        p1 = (int((bboxs[i, 0] - bboxs[i, 2] / 2) * w), int((bboxs[i, 1] - bboxs[i, 3] / 2) * h))
        p2 = (int((bboxs[i, 0] + bboxs[i, 2] / 2) * w), int((bboxs[i, 1] + bboxs[i, 3] / 2) * h))
        class_name = class_names[int(bboxs[i, 5])]
        # confidence = bboxs[i, 4]
        cv2.rectangle(img, p1, p2, color=colors[int(bboxs[i, 5])], thickness=2)
        cv2.putText(img, class_name, p1, cv2.FONT_HERSHEY_TRIPLEX, 0.9, colors[int(bboxs[i, 5])])
    return img