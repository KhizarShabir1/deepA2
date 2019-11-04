# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:31:37 2019

@author: SAN
"""

def buildTargets(pred_boxes,pred_conf,pred_cls,targets,anchors,num_anchors,num_classes,grid_size,ignore_thres,img_size):
    batch_size = targets.size(0)
    mask = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    conf_mask = torch.ones(batch_size,num_anchors,grid_size,grid_size)
    tx = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    ty = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    tw = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    th = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    tconf = torch.ByteTensor(batch_size,num_anchors,grid_size,grid_size).fill_(0)
    tcls = torch.ByteTensor(batch_size,num_anchors,grid_size,grid_size,num_classes).fill_(0)

    num_ground_truth = 0
    num_correct = 0
    for batch_idx in range(batch_size):
        for target_idx in range(targets.shape[1]):
            # there is no target, continue
            if targets[batch_idx, target_idx].sum() == 0:
                continue
            num_ground_truth += 1

            # convert to position relative to bounding box
            gx = targets[batch_idx,target_idx, 1] * grid_size
            gy = targets[batch_idx,target_idx, 2] * grid_size
            gw = targets[batch_idx,target_idx, 3] * grid_size
            gh = targets[batch_idx,target_idx, 4] * grid_size

            # get grid box indices
            gi = int(gx)
            gj = int(gy)

            '''
            get the anchor box that has the highest iou with [gw, gh]
            '''
            # shape of the gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # get iou
            anchor_iou = bboxIOU(gt_box, anchor_shapes, True)
            # ingore iou that is larger than some threshold
            conf_mask[batch_idx, anchor_iou > ignore_thres, gj, gi] = 0
            # best matching anchor box
            best = np.argmax(anchor_iou)
            '''
            calculate the best iou between target and best pred box
            '''
            # ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # best pred box
            pred_box = pred_boxes[batch_idx, best, gj, gi].type(torch.FloatTensor).unsqueeze(0)
            mask[batch_idx, best, gj, gi] = 1
            conf_mask[batch_idx, best, gj, gi] = 1

            '''
            get target box dimension that is relative to gird rather than the entire
            input image as which may come in different dimensions. Grid size is fixed.
            So we predict position that is relative to gird.
            '''
            tx[batch_idx, best, gj, gi] = gx - gi
            ty[batch_idx, best, gj, gi] = gy - gj
            tw[batch_idx, best, gj, gi] = math.log(gw / anchors[best][0] + 1e-16)
            th[batch_idx, best, gj, gi] = math.log(gh / anchors[best][1] + 1e-16)

            target_label = int(targets[batch_idx, target_idx, 0])
            tcls[batch_idx, best, gj, gi, target_label] = 1
            tconf[batch_idx, best, gj, gi] = 1

            # calculate iou
            iou = bboxIOU(gt_box, pred_box, False)
            pred_label = torch.argmax(pred_cls[batch_idx, best, gj, gi])
            score = pred_conf[batch_idx, best, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                num_correct += 1

    return num_ground_truth, num_correct, mask, conf_mask, tx, ty, tw, th, tconf, tcls

def bboxIOU(box1, box2, x1y1x2y2):
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # convert center to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    intersect_x1 = torch.max(b1_x1, b2_x1)
    intersect_y1 = torch.max(b1_y1, b2_y1)
    intersect_x2 = torch.min(b1_x2, b2_x2)
    intersect_y2 = torch.min(b1_y2, b2_y2)

    intersect_area = (intersect_x2 - intersect_x1 + 1) * (intersect_y2 - intersect_y1 + 1)

    # union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = intersect_area/(b1_area+b2_area-intersect_area+1e-16)

    return iou



loss_x = self.lambda_coord * self.mse_loss(x[mask], tx[mask])
loss_y = self.lambda_coord * self.mse_loss(y[mask], ty[mask])
loss_w = self.lambda_coord * self.mse_loss(w[mask], tw[mask])
loss_h = self.lambda_coord * self.mse_loss(h[mask], th[mask])

loss_conf = self.lambda_noobj * self.mse_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.mse_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
loss_cls = (1 / batch_size) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls