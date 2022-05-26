import torch
from torch import nn

from .processing import iou

class YoloLoss(nn.Module):
    def __init__(self, S, B):
        super().__init__()
        self.S = S
        self.B = B

    def forward(self, preds, targets):
        batch_size = targets.size(0)

        loss_coord_xy = 0.  # coord xy loss
        loss_coord_wh = 0.  # coord wh loss
        loss_obj = 0.  # obj loss
        loss_no_obj = 0.  # no obj loss
        loss_class = 0.  # class loss

        for i in range(batch_size):
            for y in range(self.S):
                for x in range(self.S):
                    # this region has object
                    if targets[i, y, x, 4] == 1:
                        pred_bbox1 = torch.Tensor(
                            [preds[i, y, x, 0], preds[i, y, x, 1], preds[i, y, x, 2], preds[i, y, x, 3]])
                        pred_bbox2 = torch.Tensor(
                            [preds[i, y, x, 5], preds[i, y, x, 6], preds[i, y, x, 7], preds[i, y, x, 8]])
                        target_bbox = torch.Tensor(
                            [targets[i, y, x, 0], targets[i, y, x, 1], targets[i, y, x, 2], targets[i, y, x, 3]])

                        # compute iou of two bbox
                        iou1 = iou(pred_bbox1, target_bbox)
                        iou2 = iou(pred_bbox2, target_bbox)

                        # judge responsible box
                        if iou1 > iou2:
                            # calculate coord xy loss
                            loss_coord_xy += 5 * torch.sum((targets[i, y, x, 0:2] - preds[i, y, x, 0:2]) ** 2)

                            # coord wh loss
                            loss_coord_wh += torch.sum((targets[i, y, x, 2:4].sqrt() - preds[i, y, x, 2:4].sqrt()) ** 2)

                            # obj confidence loss
                            loss_obj += (iou1 - preds[i, y, x, 4]) ** 2

                            # no obj confidence loss
                            loss_no_obj += 0.5 * ((0 - preds[i, y, x, 9]) ** 2)
                        else:
                            # coord xy loss
                            loss_coord_xy += 5 * torch.sum((targets[i, y, x, 5:7] - preds[i, y, x, 5:7]) ** 2)

                            # coord wh loss
                            loss_coord_wh += torch.sum((targets[i, y, x, 7:9].sqrt() - preds[i, y, x, 7:9].sqrt()) ** 2)

                            # obj confidence loss
                            loss_obj += (iou2 - preds[i, y, x, 9]) ** 2

                            # no obj confidence loss
                            loss_no_obj += 0.5 * ((0 - preds[i, y, x, 4]) ** 2)

                        # class loss
                        loss_class += torch.sum((targets[i, y, x, 10:] - preds[i, y, x, 10:]) ** 2)

                    # this region has no object
                    else:
                        loss_no_obj += 0.5 * torch.sum((0 - preds[i, y, x, [4, 9]]) ** 2)

        return (loss_coord_xy + loss_coord_wh + loss_obj + loss_no_obj + loss_class) / batch_size   # loss 
