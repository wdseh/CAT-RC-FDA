import json
import os
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

def get_bdd100k_dicts(json_file):
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
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": anno["category_id"],
                "iscrowd": anno["iscrowd"],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def register_bdd100k_dataset(json_file, image_root, dataset_name):
    DatasetCatalog.register(dataset_name, lambda: get_bdd100k_dicts(json_file))
    MetadataCatalog.get(dataset_name).set(thing_classes=['person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'])

# Replace these paths with your actual paths
json_file_train = "./datasets/bdd100k_voc/train.json"
json_file_val = "./datasets/bdd100k_voc/val.json"
image_root = "./datasets/bdd100k_voc/JPEGImages"

# Register the datasets
register_bdd100k_dataset(json_file_train, image_root, "bdd100k_train")
register_bdd100k_dataset(json_file_val, image_root, "bdd100k_val")

# print(DatasetCatalog.list())

# Now you can use 'bdd100k_train' and 'bdd100k_val' in your config file or directly in your code
