import cv2
import numpy as np
from detectron2.structures import BoxMode, Instances
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from sklearn.cluster import DBSCAN  # 用于合并不良gt_box区域
import torch

# IoU计算函数
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0

# 合并多个gt_box为一个区域
def merge_boxes(boxes):
    x1_min = min(box[0] for box in boxes)
    y1_min = min(box[1] for box in boxes)
    x2_max = max(box[2] for box in boxes)
    y2_max = max(box[3] for box in boxes)
    return (x1_min, y1_min, x2_max, y2_max)

# 找到所有IoU < 0.5的gt_box
def get_bad_gt_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):
    bad_gt_boxes = []
    for gt_box in gt_boxes:
        is_bad = True
        for pred_box in pred_boxes:
            iou = calculate_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                is_bad = False
                break
        if is_bad:
            bad_gt_boxes.append(gt_box)
    return bad_gt_boxes

# 合并gt_box并调整区域大小
def adjust_box_size(merged_box, min_size=(300, 300), max_size=(600, 600)):
    x1, y1, x2, y2 = merged_box
    width, height = x2 - x1, y2 - y1

    if width < min_size[0] or height < min_size[1]:
        # 扩展到最小尺寸
        dx = (min_size[0] - width) // 2
        dy = (min_size[1] - height) // 2
        x1 -= dx
        y1 -= dy
        x2 += dx
        y2 += dy
    elif width > max_size[0] or height > max_size[1]:
        # 缩小到最大尺寸
        dx = (width - max_size[0]) // 2
        dy = (height - max_size[1]) // 2
        x1 += dx
        y1 += dy
        x2 -= dx
        y2 -= dy
    
    return (x1, y1, x2, y2)

# 读取和处理输入数据
def process_image_and_generate_labels(cfg, image_path, gt_boxes, pred_boxes):
    # 加载模型和图片
    with torch.no_grad():
        (
            _,
            proposals_rpn_crop,
            proposals_roih_crop,
            _,
        ) = self.model_teacher(crop_list, branch="unsup_data_weak")
        
    joint_proposal_dict = {}
    joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            
    #Process pseudo labels and thresholding
    (
        pesudo_proposals_rpn_unsup_k,
        nun_pseudo_bbox_rpn,
    ) = self.process_pseudo_label(
        proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
    )

    joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
    # Pseudo_labeling for ROI head (bbox location/objectness)
    pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
        proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
    )
    joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

    # 获取模型预测
    for item1, item2 in zip(pesudo_proposals_roih_unsup_k, unlabel_data_k):
        instances1 = item1
        pred_boxes = instances1.gt_boxes  # 预测框
        instances2 = item2["instances"]
        gt_boxes = instances2.gt_boxes

        # 第一步：筛选出IoU小于0.5的gt_box
        bad_gt_boxes = get_bad_gt_boxes(gt_boxes, pred_boxes)
        len_size = len(bad_gt_boxes)
        flag = [0] * len_size
        areas = []
                
        for i in range(len_size):
            if flag[i]:
                continue
            
            area = bad_gt_boxes[i]
            area = area.tensor.view(-1)
                
                for j in range(i + 1, len_size):
                    if flag[j]:
                        continue
                    
                    x1 = bad_gt_boxes[i][0]
                    x2 = bad_gt_boxes[i][2]
                    y1 = bad_gt_boxes[i][1]
                    y2 = bad_gt_boxes[i][3]
                    new_x1 = min(x1, area[0])
                    new_x2 = max(x2, area[2])
                    new_y1 = min(y1, area[1])
                    new_y2 = max(y2, area[3])
                    
                    if (new_x2 - new_x1) > 600 or (new_y2 - new_y1) > 300:
                        continue
                    
                    area[0] = new_x1
                    area[1] = new_x2
                    area[2] = new_y1
                    area[3] = new_y2
                    
                    flag[j] = 1
                    
            areas.append(area)
        
        
        for item in areas:
            if item[1] - item[0] < 600:
                gap = 600 - (item[1] - item[0])
                item[0] = item[0] - (gap / 2)
                item[1] = item[1] + (gap / 2)
                if item[0] < 0:
                    item[1] = item[1] - item[0]
                    item[0] = 0
                if item[1] > 600:
                    item[0] = item[0] - (item[1] - 600)
                    item[1] = 600
                    
            if item[3] - item[2] < 300:
                gap = 300 - (item[3] - item[2])
                item[2] = item[2] - (gap / 2)
                item[3] = item[3] + (gap / 2)
                if item[2] < 0:
                    item[3] = item[3] - item[2]
                    item[2] = 0
                if item[3] > 300:
                    item[2] = item[2] - (item[3] - 300)
                    item[3] = 300
                    
        for item in areas:
            

        # 如果有合并区域，则裁剪原图
        if adjusted_box:
            x1, y1, x2, y2 = adjusted_box
            # 从长宽为原图2倍的区域中裁剪对应区域
            expanded_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
            cropped_image = expanded_image[max(0, y1):min(expanded_image.shape[0], y2),
                                        max(0, x1):min(expanded_image.shape[1], x2)]
            
            # 保存裁剪图像
            cv2.imwrite("cropped_bad_region.jpg", cropped_image)

            # 可视化合并后的区域
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
            v.draw_box(adjusted_box, color=(255, 0, 0), thickness=3)
            output_image = v.get_output().get_image()
            cv2.imwrite("visualized_adjusted_region.jpg", output_image)

            print("Merged and adjusted box:", adjusted_box)
            return cropped_image
        else:
            print("No bad regions found to merge.")
            return None

# 使用教师网络进行伪标签融合（伪代码，需根据具体实现）
def fuse_with_teacher_network(cropped_image, teacher_network_predictor):
    teacher_output = teacher_network_predictor(cropped_image)
    # 使用融合算法（例如加权平均或NMS）融合伪标签
    # 在这里调用你师兄给的代码来进行伪标签融合
    return teacher_output
