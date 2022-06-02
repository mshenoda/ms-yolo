import numpy as np
import torch 
from collections import Counter
from utils import iou, decode
import matplotlib.pyplot as plt
# def average_precision(preds, targets, iou_thresh=0.5, conf_thresh=0.1):
#     assert len(preds) == len(targets)

#     tp = np.zeros(len(targets))
#     fp = np.zeros(len(targets))
#     for i in range(len(targets)):
#         if preds[i][5] > conf_thresh:
#             current_iou = iou(preds[i], targets[i])
#             if current_iou > iou_thresh:
#                 tp[i] = 1
#             else:
#                 fp[i] = 1
#     tp = np.sum(tp)
#     fp = np.sum(fp)

#     ap = tp / (tp + fp)

#     return ap


def mean_average_precision(preds, labels, S, B, num_classes=20):
    thresholds = np.arange(start=0.1, stop=0.5, step=0.2)
    avg_precision = 0.5
    avg_precision_count = 1
    for iou_threshold in thresholds:
        pred_boxes = decode(preds, S, B, num_classes, 0.1, iou_threshold).tolist()
        true_boxes = labels.tolist()[0]

        # list storing all AP for respective classes
        current_precisions = list()

        # used for numerical stability later on
        epsilon = 1e-6
        
        for c in range(num_classes):
            detections = list()
            ground_truths = list()

            # Go through all predictions and targets,
            # and only add the ones that belong to the
            # current class c
            for detection in pred_boxes:
                if detection[5] == c:
                    detections.append(detection)

            for true_box in true_boxes:
                if true_box[5] == c:
                    ground_truths.append(true_box)

            # find the amount of bboxes for each training example
            # Counter here finds how many ground truth bboxes we get
            # for each training example, so let's say img 0 has 3,
            # img 1 has 5 then we will obtain a dictionary with:
            # amount_bboxes = {0:3, 1:5}
            amount_bboxes = Counter([gt[5] for gt in ground_truths])

            # We then go through each key, val in this dictionary
            # and convert to the following (w.r.t same example):
            # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            # sort by box probabilities which is index 4
            detections.sort(key=lambda x: x[5], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)
            
            # If none exists for this class then we can safely skip
            if total_true_bboxes == 0:
                continue

            for detection_idx, detection in enumerate(detections):
                # Only take out the ground_truths that have the same
                # training idx as detection
                ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[5] == detection[5]
                ]

                num_gts = len(ground_truth_img)
                best_iou = 0

                for idx, gt in enumerate(ground_truth_img):
                    curr_iou = iou(
                        torch.tensor(detection[0:4]),
                        torch.tensor(gt[0:4])
                    )

                    if curr_iou > best_iou:
                        best_iou = curr_iou
                        best_gt_idx = idx

                if best_iou > iou_threshold:
                    # only detect ground truth detection once
                    if amount_bboxes[detection[5]][best_gt_idx] == 0:
                        # true positive and add this bounding box to seen
                        TP[detection_idx] = 1
                        amount_bboxes[detection[5]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # torch.trapz for numerical integration
            current_precisions.append(torch.trapz(precisions, recalls))
            
            if len(current_precisions) > 0:
                avg_precision += ((sum(current_precisions) / float(len(current_precisions))).item())
                avg_precision_count += 1
    
    return (avg_precision / avg_precision_count)*100.0