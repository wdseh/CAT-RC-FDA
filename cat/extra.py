# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators
# from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators

from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

from cat.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from cat.data.dataset_mapper import DatasetMapperTwoCropSeparate, DatasetMapperTwoCropSeparate_2
from cat.engine.hooks import LossEvalHook
from cat.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from cat.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from cat.solver.build import build_lr_scheduler
from cat.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator
from cat.icrm import ICRm

from .probe import OpenMatchTrainerProbe
import copy

import torch
from PIL import Image
import matplotlib.pyplot as plt

import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.nn.functional import interpolate

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
from detectron2.structures import Boxes
import cv2


def load_and_resize_image(file_name, output_size=(1200, 2400)):
    image = Image.open(file_name).convert("RGB")
    image = transforms.functional.to_tensor(image)  # 转换为 PyTorch 张量
    image_resized = interpolate(image.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False).squeeze(0)
    return image_resized




# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 16

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


# CAT Trainer
class CATTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        print("begin of dataloader")
        data_loader = self.build_train_loader(cfg)
        print("end of dataloader")

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.probe = OpenMatchTrainerProbe(cfg)
        self.icrm = ICRm(cfg.MODEL.ROI_HEADS.NUM_CLASSES, 50, cfg.OUTPUT_DIR, blocked_classes=self.cfg.BLOCKED_CLASSES,mix_ratio = self.cfg.MIX_RATIO, cfg = self.cfg)
        self.register_hooks(self.build_hooks()) 
        self.top_eval_ap = 0.0
        self.top_eval_ap_stu = 0.0

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        # print("cfg.MODEL.WEIGHTS:")
        # print(self.cfg.MODEL.WEIGHTS, resume)
        
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        print("begin of mapper")
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        mapper_2 = DatasetMapperTwoCropSeparate_2(cfg, True)
        print("end of mapper")
        return build_detection_semisup_train_loader_two_crops(cfg, mapper, mapper_2)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)

        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            print(self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            full_scores = proposal_bbox_inst.full_scores.clone()

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)
            
            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
            new_proposal_inst.full_scores = full_scores[valid_map]

        return new_proposal_inst
    
    

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        
        return label_list
    
    # def get_label_test(self, label_data):
    #     label_list = []
    #     for label_datum in label_data:
    #         if "instances" in label_datum.keys():
    #             label_list.append(label_datum["instances"])

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q_expand, unlabel_data_k_expand = data
        
        # if comm.is_main_process():
        #     print(len(unlabel_data_k_expand))
        #     print(unlabel_data_k_expand[0])
        #     print(unlabel_data_k_expand[0]['image'].shape)
            # exit()
        #     print(len(label_data_q))
        #     print(type(label_data_q[0]['image']))
        #     print(label_data_q[0]['image'])
            # print(label_data_q[0]['image'].shape)
            # exit()
            
        data_time = time.perf_counter() - start
        
        if self.cfg.MIXUP:
            with torch.no_grad():
                self.icrm.save_crops(label_data_k) 
                label_data_k = self.icrm.mix_crop_new(label_data_k)
                label_data_q = self.icrm.add_labels(label_data_q)
        else:
            with torch.no_grad():
                self.icrm.save_crops(label_data_k) 
                label_data_k = self.icrm.add_labels(label_data_k)
                label_data_q = self.icrm.add_labels(label_data_q)
            
        if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
            # update copy the the whole model
            self._update_teacher_model(keep_rate=0.00)
            # self.model.build_discriminator()

        elif (
            self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
        ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0: # and self.iter >= self.cfg.SEMISUPNET.BURN_UP_STEP:
            self._update_teacher_model(
                keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            if self.cfg.CLS_LOSS:
                class_info = self.icrm.class_info
            else:
                class_info = None
            # input both strong and weak supervised data into model
            label_data_k.extend(label_data_q)
            record_dict, _, _, proposals_predictions = self.model(
                label_data_k, branch="supervised",class_info = class_info)
                
            with torch.no_grad():
                self.icrm.get_matches(proposals_predictions,self.iter)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:

            record_dict = {}

            ######################## For probe #################################
            # import pdb; pdb. set_trace() 
            # gt_unlabel_k = self.get_label(unlabel_data_k)
            # gt_unlabel_q = self.get_label_test(unlabel_data_q)
            

            #  0. remove unlabeled data labels
            unlabel_data_q_expand = self.remove_label(unlabel_data_q_expand)
            unlabel_data_k_expand = self.remove_label(unlabel_data_k_expand)
            
            # 更新 unlabel_data_k 中的图像
            unlabel_data_k = []
            unlabel_data_q = []
            resize_transform = transforms.Resize(size=(600, 1200))
            for data in unlabel_data_k_expand:
                # 创建data的深拷贝，这样原始data不会被修改
                data_copy = data.copy()
                
                # 提取文件名，并加载调整大小后的图像
                # file_name = data_copy['file_name']
                # image_tensor = load_and_resize_image(file_name, output_size=(1200, 2400))
                image_tensor = resize_transform(data_copy['image'])
                
                # image_tensor = image_tensor * 255  # 将值缩放到 [0, 255]
                image_tensor = image_tensor.to(torch.uint8)
                
                # 替换原有的'image'键对应的值
                data_copy['image'] = image_tensor
                
                # 将修改后的字典添加到新的列表中
                unlabel_data_k.append(data_copy)
                
            for data in unlabel_data_q_expand:
                data_copy = data.copy()
                image_tensor = resize_transform(data_copy['image'])
                image_tensor = image_tensor.to(torch.uint8)
                data_copy['image'] = image_tensor
                unlabel_data_q.append(data_copy)
            
            # if comm.is_main_process():  
            #     print("image_size", label_data_k[0]['image'].shape)
            #     print("image_instance", type(label_data_k[0]), label_data_k[0])
                
                
            #     image_tensor = label_data_k[0]['image']
            #     image_pil = transforms.ToPILImage()(image_tensor)
            #     # 如果你想要保存图像到文件
            #     image_pil.save('input_image.jpg')
                
            #     print("image_size", label_data_q[0]['image'].shape)
            #     print("image_instance", type(label_data_q[0]), label_data_q[0])
                
                
            #     image_tensor = label_data_q[0]['image']
            #     image_pil = transforms.ToPILImage()(image_tensor)
            #     # 如果你想要保存图像到文件
            #     image_pil.save('input_image_strong.jpg')
                
            #     print("image_size", unlabel_data_k[0]['image'].shape)
            #     print("image_instance", type(unlabel_data_k[0]), unlabel_data_k[0])
                
                
            #     image_tensor = unlabel_data_k[0]['image']
            #     image_pil = transforms.ToPILImage()(image_tensor)
            #     # 如果你想要保存图像到文件
            #     image_pil.save('output_image.jpg')
                
            #     print("image_size_q", unlabel_data_q[0]['image'].shape)
            #     print("image_instance_q", type(unlabel_data_q[0]), unlabel_data_q[0])
                
                
            #     image_tensor = unlabel_data_q[0]['image']
            #     image_pil = transforms.ToPILImage()(image_tensor)
            #     # 如果你想要保存图像到文件
            #     image_pil.save('output_image_strong.jpg')
                
            #     print("image_size_expand", unlabel_data_k_expand[0]['image'].shape)
            #     print("image_instance_expand", type(unlabel_data_k_expand[0]), unlabel_data_k_expand[0])
                
            #     image_tensor = unlabel_data_k_expand[0]['image']
            #     image_pil = transforms.ToPILImage()(image_tensor)
            #     # 如果你想要保存图像到文件
            #     image_pil.save('output_image_expand.jpg')
            #     exit()
            #  1. generate the pseudo-label using teacher model
            
                
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k_expand, branch="unsup_data_weak")
            
            # if comm.is_main_process():  
            #     print("propo_rpn:", proposals_rpn_unsup_k[0])
            #     print("propo_roih:", proposals_roih_unsup_k[0])
            #     print("propo_rpn_expand:", proposals_rpn_unsup_k_expand[0])
            #     print("propo_roih_expand:", proposals_roih_unsup_k_expand[0])
            # proposals_rpn_unsup_k_expand.proposal_boxes.tensor = proposals_rpn_unsup_k_expand.proposal_boxes.tensor / 2
            
            for item in proposals_rpn_unsup_k:
                # print("item_1", item)
                item.proposal_boxes.tensor = item.proposal_boxes.tensor / 2
                item._image_size = (600, 1200)
                
            for item in proposals_roih_unsup_k:
                # print("item_2", item)
                item.pred_boxes.tensor = item.pred_boxes.tensor / 2
                item._image_size = (600, 1200)
            
            # if comm.is_main_process():  
            #     print("propo:", proposals_rpn_unsup_k[0])
                # print("rpn_unsup_k", type(proposals_rpn_unsup_k), len(proposals_rpn_unsup_k), proposals_rpn_unsup_k[0])
                # print("roih_unsup_k", type(proposals_roih_unsup_k), len(proposals_roih_unsup_k), proposals_roih_unsup_k[0])
                # print("rpn_unsup_k_expand", type(proposals_rpn_unsup_k_expand), len(proposals_rpn_unsup_k_expand), proposals_rpn_unsup_k_expand[1])
                # print("roih_unsup_k_expand", type(proposals_roih_unsup_k_expand), len(proposals_roih_unsup_k_expand), proposals_roih_unsup_k_expand[1])
                
                # proposals_rpn_unsup_k_expand[0].proposal_boxes.tensor = proposals_rpn_unsup_k_expand[0].proposal_boxes.tensor / 2
                
                # print("rpn_unsup_k_expand", type(proposals_rpn_unsup_k_expand), len(proposals_rpn_unsup_k_expand), proposals_rpn_unsup_k_expand[0])
                
                # print("type", type(proposals_rpn_unsup_k_expand[0]))
                # exit()

            #  2. Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

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

            # 3. add pseudo-label to unlabeled data

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )
            
            
            if comm.is_main_process() and (self.iter + 1) % 1000 == 0:
            
                unlabel_data_q_2 = unlabel_data_q.copy()
                unlabel_data_k_2 = unlabel_data_k.copy()
            
                with torch.no_grad():
                    (
                        _,
                        proposals_rpn_unsup_k_expand,
                        proposals_roih_unsup_k_expand,
                        _,
                    ) = self.model_teacher(unlabel_data_k_2, branch="unsup_data_weak")
                
                joint_proposal_dict = {}
                joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k_expand
                #Process pseudo labels and thresholding
                (
                    pesudo_proposals_rpn_unsup_k_expand,
                    nun_pseudo_bbox_rpn,
                ) = self.process_pseudo_label(
                    proposals_rpn_unsup_k_expand, cur_threshold, "rpn", "thresholding"
                )

                joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k_expand
                # Pseudo_labeling for ROI head (bbox location/objectness)
                pesudo_proposals_roih_unsup_k_expand, _ = self.process_pseudo_label(
                    proposals_roih_unsup_k_expand, cur_threshold, "roih", "thresholding"
                )
                joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k_expand
                
                unlabel_data_q_2 = self.add_label(
                    unlabel_data_q_2, joint_proposal_dict["proposals_pseudo_roih"]
                )
                unlabel_data_k_2 = self.add_label(
                    unlabel_data_k_2, joint_proposal_dict["proposals_pseudo_roih"]
                )
                
                
                image_np = unlabel_data_k[0]['image'].permute(1, 2, 0).cpu().numpy()
                for box, cls in zip(unlabel_data_k[0]['instances'].gt_boxes, unlabel_data_k[0]['instances'].gt_classes):
                    # 生成随机颜色
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    
                    x0, y0, x1, y1 = map(int, box)
                    image_np = image_np.copy()
                    cv2.rectangle(image_np, (x0, y0), (x1, y1), color, 2)  # 绘制边界框
                    # cv2.putText(image_np, cls.item(), (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imwrite('un_k_' + str(self.iter) + '.jpg', image_np)
               
                image_np = unlabel_data_q[0]['image'].permute(1, 2, 0).cpu().numpy()
                for box, cls in zip(unlabel_data_q[0]['instances'].gt_boxes, unlabel_data_q[0]['instances'].gt_classes):
                    # 生成随机颜色
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    
                    x0, y0, x1, y1 = map(int, box)
                    image_np = image_np.copy()
                    cv2.rectangle(image_np, (x0, y0), (x1, y1), color, 2)  # 绘制边界框
                    # cv2.putText(image_np, cls.item(), (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imwrite('un_q_' + str(self.iter) + '.jpg', image_np)
                
                image_np = unlabel_data_k_2[0]['image'].permute(1, 2, 0).cpu().numpy()
                for box, cls in zip(unlabel_data_k_2[0]['instances'].gt_boxes, unlabel_data_k_2[0]['instances'].gt_classes):
                    # 生成随机颜色
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    
                    x0, y0, x1, y1 = map(int, box)
                    image_np = image_np.copy()
                    cv2.rectangle(image_np, (x0, y0), (x1, y1), color, 2)  # 绘制边界框
                    # cv2.putText(image_np, cls.item(), (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imwrite('un_k_2_' + str(self.iter) + '.jpg', image_np)
                
                image_np = unlabel_data_q_2[0]['image'].permute(1, 2, 0).cpu().numpy()
                for box, cls in zip(unlabel_data_q_2[0]['instances'].gt_boxes, unlabel_data_q_2[0]['instances'].gt_classes):
                    # 生成随机颜色
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    
                    x0, y0, x1, y1 = map(int, box)
                    image_np = image_np.copy()
                    cv2.rectangle(image_np, (x0, y0), (x1, y1), color, 2)  # 绘制边界框
                    # cv2.putText(image_np, cls.item(), (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imwrite('un_q_2_' + str(self.iter) + '.jpg', image_np)
                
                # exit()
            
            # 3. add pseudo-label to unlabeled data
            
            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

            # 3. add pseudo-label to unlabeled data

            if self.cfg.MIXUP:
                with torch.no_grad():
                    self.icrm.save_crops_target(unlabel_data_k)
                    all_unlabel_data = self.icrm.mix_crop_new(all_unlabel_data, True)
            else:
                with torch.no_grad():
                    self.icrm.save_crops_target(unlabel_data_k)
                    all_unlabel_data = self.icrm.add_labels(all_unlabel_data)

            if self.cfg.CLS_LOSS:
                class_info = self.icrm.class_info
            else:
                class_info = None

            # 4. input both strongly and weakly augmented labeled data into student model
            record_all_label_data, _, _, proposals_predictions = self.model(
                all_label_data, branch="supervised", class_info = class_info
            )
            record_dict.update(record_all_label_data)

            with torch.no_grad():
                self.icrm.get_matches(proposals_predictions,self.iter)
    


            if self.cfg.CLS_LOSS:
                if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP+400:
                    class_info = self.icrm.class_info
                else:
                    class_info = self.icrm.target_class_info
            else:
                class_info = None
            

            # 5. input strongly augmented unlabeled data into model
            record_all_unlabel_data, _, _, proposals_predictions_unlabel = self.model(
                all_unlabel_data, branch="supervised_target", class_info = class_info
            )
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            with torch.no_grad():
                self.icrm.get_matches(proposals_predictions_unlabel,self.iter, True)

            # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
            # give sign to the target data

            for i_index in range(len(unlabel_data_k)):
                for k, v in unlabel_data_k[i_index].items():
                    label_data_k[i_index][k + "_unlabeled"] = v

            all_domain_data = label_data_k
            record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="domain")
            record_dict.update(record_all_domain_data)


            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0 already set to 0 in loss weights, used for bbox paste
                        if self.cfg.TARGET_BBOX:
                            loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT #0
                        else:
                            loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] *
                            self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    elif (
                        key == "loss_D_img_s" or key == "loss_D_img_t"
                    ):  # set weight for discriminator
                        # import pdb
                        # pdb.set_trace()
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        #self.clip_gradient(self.model, 10.)
        self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def test_eval(self):
        if comm.is_main_process():
            print("teacher_test")
        _last_eval_results_teacher = self.test(self.cfg, self.model_teacher)
        
        if comm.is_main_process():
            print("teacher_result")
            print(type(_last_eval_results_teacher), _last_eval_results_teacher)
            
        if comm.is_main_process():
            print("student_test")
        _last_eval_results_student = self.test(self.cfg, self.model)
        
        if comm.is_main_process():
            print("student_result")
            print(type(_last_eval_results_student), _last_eval_results_student)
        
    
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    def _update_student_model(self, type='full_copy', keep_rate=0.9996, rand_rate = 1.0):
        teacher_model_dict = self.model_teacher.state_dict()

        new_student_dict = OrderedDict()
        for key, value in self.model.state_dict().items():
            if comm.get_world_size() > 1:
                teacher_key = key[7:]
            else:
                teacher_key = key

            if teacher_key in teacher_model_dict.keys():
                
                if type == 'full_copy':
                    new_student_dict[key] = (
                        teacher_model_dict[teacher_key] *
                        (1 - keep_rate) + value * keep_rate
                    )
                elif type == 'random_copy':
                    if np.random.rand() < rand_rate:
                        new_student_dict[key] = (
                            teacher_model_dict[teacher_key] *
                            (1 - keep_rate) + value * keep_rate
                        )
                    else:
                        new_student_dict[key] = value
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model.load_state_dict(new_student_dict)



    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 16  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def clip_gradient(self, model, clip_norm):
        """Computes a gradient clipping coefficient based on gradient norm."""
        totalnorm = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                try:
                    modulenorm = p.grad.norm()
                except:
                    continue
                totalnorm += modulenorm ** 2
        totalnorm = torch.sqrt(totalnorm).item()
        norm = (clip_norm / max(totalnorm, clip_norm))
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.mul_(norm)