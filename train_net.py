#!/usr/bin/env python3

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from cat import add_cat_config
from cat.engine.trainer import CATTrainer

# hacky way to register
import cat.data.datasets.builtin
from cat.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from cat.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from cat.modeling.proposal_generator.rpn import PseudoLabRPN
from cat.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from cat.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

import os
import json
import os
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

def get_dataset_dicts(json_file):
    with open(json_file) as f:
        img_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(img_anns['images']):
        record = {}
        
        filename = v["file_name"]
        height = v["height"]
        width = v["width"]
        
        record["file_name"] = filename
        record["image_id"] = v["id"]
        record["height"] = height
        record["width"] = width

        annos = [a for a in img_anns["annotations"] if a["image_id"] == v["id"]]
        objs = []
        for anno in annos:
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": anno["category_id"],
                "iscrowd": anno["iscrowd"],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def register_dataset(json_file, image_root, dataset_name):
    DatasetCatalog.register(dataset_name, lambda: get_dataset_dicts(json_file))
    MetadataCatalog.get(dataset_name).set(thing_classes=["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"])

def register_dataset_val(json_file, image_root, dataset_name):
    DatasetCatalog.register(dataset_name, lambda: get_dataset_dicts(json_file))
    MetadataCatalog.get(dataset_name).set(
        thing_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
        evaluator_type="coco"  # 注意：这里使用的是字符串 "coco"，而不是 COCOEvaluator 类
    )


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_cat_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.INPUT.CROP.ENABLED = True
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)    
    # Replace these paths with your actual paths
    # json_file_train = "./datasets/bdd100k_voc/train.json"
    # json_file_val = "./datasets/bdd100k_voc/val.json"
    # image_root = "./datasets/bdd100k_voc/JPEGImages"

    # # Register the datasets
    # register_dataset(json_file_train, image_root, "bdd100k_train")
    # register_dataset(json_file_val, image_root, "bdd100k_val")

    json_file_train = "./datasets/voc_all/trainval_2012_with_xy.json"
    json_file_train_2 = "./datasets/clipart/train_with_xy.json"
    json_file_val = "./datasets/clipart/test_with_xy.json"

    image_root_train = "./datasets/voc_all/JPEGImages_2012"
    image_root_train_2 = "./datasets/clipart/JPEGImages"
    image_root_val = "./datasets/clipart/JPEGImages"

    # Register the datasets
    register_dataset(json_file_train, image_root_train, "pascal_voc_train")
    register_dataset(json_file_train_2, image_root_train_2, "clipart_train")
    register_dataset_val(json_file_val, image_root_val, "clipart_val")

    # print(DatasetCatalog.list())

    if cfg.SEMISUPNET.Trainer == "cat":
        print(1)
        Trainer = CATTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
