import numpy as np
import torch
from collections import Counter

def encode(bboxes, S, B, num_classes):
    # bbox is cx, cy, width, height, classification
    label = np.zeros((S, S, 5 * B + num_classes))
    for x, y, w, h, c in bboxes:
        x_grid = int(x // (1.0 / S))
        y_grid = int(y // (1.0 / S))
        xx, yy = x, y
        label[y_grid, x_grid, 0:5] = np.array([xx, yy, w, h, 1])
        label[y_grid, x_grid, 5:10] = np.array([xx, yy, w, h, 1])
        label[y_grid, x_grid, 10 + c] = 1
    return label


def decode(pred, S, B, num_classes, conf_thresh, iou_thresh):
    bboxes = torch.zeros((S * S, 5 + num_classes))  # 49*25
    for x in range(S):
        for y in range(S):
            conf1, conf2 = pred[x, y, 4], pred[x, y, 9]
            if conf1 > conf2:
                # bbox1
                bboxes[(x * S + y), 0:4] = torch.Tensor([pred[x, y, 0], pred[x, y, 1], pred[x, y, 2], pred[x, y, 3]])
                bboxes[(x * S + y), 4] = pred[x, y, 4]
                bboxes[(x * S + y), 5:] = pred[x, y, 10:]
            else:
                # bbox2
                bboxes[(x * S + y), 0:4] = torch.Tensor([pred[x, y, 5], pred[x, y, 6], pred[x, y, 7], pred[x, y, 8]])
                bboxes[(x * S + y), 4] = pred[x, y, 9]
                bboxes[(x * S + y), 5:] = pred[x, y, 10:]

    # apply NMS to all bounding boxes
    decoded_bbox = nms(bboxes, num_classes, conf_thresh, iou_thresh)
    return decoded_bbox # x y w h c

def iou(bbox1, bbox2):
    # bbox: x y w h
    bbox1, bbox2 = bbox1.cpu().detach().numpy().tolist(), bbox2.cpu().detach().numpy().tolist()

    area1 = bbox1[2] * bbox1[3]  # bbox1's area
    area2 = bbox2[2] * bbox2[3]  # bbox2's area

    max_left = max(bbox1[0] - bbox1[2] / 2, bbox2[0] - bbox2[2] / 2)
    min_right = min(bbox1[0] + bbox1[2] / 2, bbox2[0] + bbox2[2] / 2)
    max_top = max(bbox1[1] - bbox1[3] / 2, bbox2[1] - bbox2[3] / 2)
    min_bottom = min(bbox1[1] + bbox1[3] / 2, bbox2[1] + bbox2[3] / 2)

    if max_left >= min_right or max_top >= min_bottom:
        return 0
    else:
        # iou = intersect / union
        intersect = (min_right - max_left) * (min_bottom - max_top)
        return intersect / (area1 + area2 - intersect)

def nms(bboxes, num_classes, conf_thresh=0.1, iou_thresh=0.3):
    # Non-Maximum Suppression
    bbox_prob = bboxes[:, 5:].clone().detach()
    bbox_conf = bboxes[:, 4].clone().detach().unsqueeze(1).expand_as(bbox_prob)
    class_conf = bbox_conf * bbox_prob 
    class_conf[class_conf <= conf_thresh] = 0

    # for each class, sort the class confidence
    for c in range(num_classes):
        rank = torch.sort(class_conf[:, c], descending=True).indices 
        # for each bbox
        for i in range(bboxes.shape[0]):
            if class_conf[rank[i], c] == 0:
                continue
            for j in range(i + 1, bboxes.shape[0]):
                if class_conf[rank[j], c] != 0:
                    curr_iou = iou(bboxes[rank[i], 0:4], bboxes[rank[j], 0:4])
                    if curr_iou > iou_thresh:
                        class_conf[rank[j], c] = 0

    # exclude class confidence score equals to 0
    bboxes = bboxes[torch.max(class_conf, dim=1).values > 0]

    class_conf = class_conf[torch.max(class_conf, dim=1).values > 0]

    ret = torch.ones((bboxes.size()[0], 6))

    # return null
    if bboxes.size()[0] == 0:
        return torch.tensor([])

    # bbox coord
    ret[:, 0:4] = bboxes[:, 0:4]
    # bbox class confidence
    ret[:, 4] = torch.max(class_conf, dim=1).values
    # bbox class
    ret[:, 5] = torch.argmax(bboxes[:, 5:], dim=1).int()
    return ret

