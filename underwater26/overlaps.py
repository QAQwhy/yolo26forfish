# Derived from the supplied project. Ultralytics AGPL-3.0; see LICENSE.
import math
import torch

def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool=True, GIoU: bool=False, DIoU: bool=False, CIoU: bool=False, CwaterIoU: bool=False, WiseIoU: bool=False, eps: float=1e-07) -> torch.Tensor:
    """
    Calculate IoU / GIoU / DIoU / CIoU / CwaterIoU.
    Enhanced CwaterIoU with MPDIoU-style width-height penalty.
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = (box1.chunk(4, -1), box2.chunk(4, -1))
        w1_, h1_ = (w1 / 2, h1 / 2)
        w2_, h2_ = (w2 / 2, h2 / 2)
        b1_x1, b1_x2 = (x1 - w1_, x1 + w1_)
        b1_y1, b1_y2 = (y1 - h1_, y1 + h1_)
        b2_x1, b2_x2 = (x2 - w2_, x2 + w2_)
        b2_y1, b2_y2 = (y2 - h2_, y2 + h2_)
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1 = b1_x2 - b1_x1 + eps
        h1 = b1_y2 - b1_y1 + eps
        w2 = b2_x2 - b2_x1 + eps
        h2 = b2_y2 - b2_y1 + eps
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU or CwaterIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU or CwaterIoU:
            c2 = cw.pow(2) + ch.pow(2) + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
            if CIoU:
                v = 4 / math.pi ** 2 * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            elif CwaterIoU:
                v = 4 / math.pi ** 2 * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                dist_weight = (1 - iou.detach()) ** 2
                w_diff = (w1 - w2).pow(2)
                h_diff = (h1 - h2).pow(2)
                mpd_penalty = 0.5 * (w_diff + h_diff) / c2
                return iou - (dist_weight * rho2 / c2 + v * alpha + mpd_penalty)
            return iou - rho2 / c2
        elif WiseIoU:
            dist_weight = torch.exp(-(1 - iou))
            return iou - dist_weight * rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou
