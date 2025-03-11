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

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_cat_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "cat":
        print(1)
        Trainer = CATTrainer
    else:
        raise ValueError("Trainer Name is not found.")


    trainer = Trainer(cfg)
    
    
    # 假设您的模型是model，并且已经通过resume_or_load函数加载了权重
    # 以下是打印模型特定层参数的代码
    print("before_load:")
    # 获取模型的状态字典
    state_dict = trainer.model.state_dict()

    # 选择您想要查看的层的名字
    layer_name = 'module.roi_heads.box_head.fc1.weight'  # 请替换为实际的层名

    # 打印该层的参数
    if layer_name in state_dict:
        print(f"Parameters of layer '{layer_name}':")
        print(state_dict[layer_name])
    else:
        print(f"Layer '{layer_name}' not found in the model.")

    
    trainer.resume_or_load(resume=args.resume)
    
    print("after_load:")
    state_dict = trainer.model.state_dict()

    # 选择您想要查看的层的名字
    layer_name = 'module.roi_heads.box_head.fc1.weight'  # 请替换为实际的层名

    # 打印该层的参数
    if layer_name in state_dict:
        print(f"Parameters of layer '{layer_name}':")
        print(state_dict[layer_name])
    else:
        print(f"Layer '{layer_name}' not found in the model.")

    trainer.rereason()


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
