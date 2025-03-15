import os
import copy
import shutil
import math
import json

import pandas as pd
from tqdm import tqdm

folder = "Datasets/Generate/Gen6/" #"Datasets/Generate/Gen4/" #"Datasets/InputIMG_02/"
df = pd.read_csv(folder + 'annotation.csv')
df[["x_right", "x_left", "y_down", "y_up"]] = df[["x_right", "x_left", "y_down", "y_up"]].astype(float)


# for Gen
#df["x_right"] += df["x_right"] - df["x_left"]
#df["y_down"] += df["y_down"] - df["y_up"]
#print(df.to_string())

"""
"images":[
        {
            "license": 6,
            "file_name": "000000466319.jpg",
            "coco_url": "http://images.cocodataset.org/test2017/000000466319.jpg",
            "height": 480,
            "width": 640,
            "date_captured": "2013-11-14 11:04:33",
            "id": 466319
        },
        ]
"annotations":[
        {
            "id": int,
            "image_id": int,
            "category_id": int,
            "segmentation": RLE or [polygon],
            "area": float,
            "bbox": [x,y,width,height],
            "iscrowd": 0 or 1,
        },
    ],
categories[
        {
            "id": int,
            "name": str,
            "supercategory": str,
        },
    ]

"""


cocoDict = {
    "info": {
        "description": "Dataset_03 - umela data", #"Dataset_04 - semi - realna data", #"InputIMG_02",
        "url": "",
        "version": "1.0",
        "year": 2024,
        "contributor": "BUT staff",
        "date_created": "2024/10/23"
    },
    "images": [],
    "annotations":[],
    "categories":[        
        #{
        #    "id": 0,
        #    "name": "Sensor1"
        #},
        #{
        #    "id": 1,
        #    "name": "Sensor2"
        #},
        #{
        #    "id": 2,
        #    "name": "Sensor3"
        #},
        #{
        #    "id": 3,
        #    "name": "Sensor4"
        #},
        #{
        #    "id": 4,
        #    "name": "Sensor5"
        #}
        {
            "id": 5,
            "name": "detector",
            "supercategory": "none",
        }
    ]
    } 

id = 2
for rowId in tqdm(df.index):
    cocoDict["images"].append(
        {
            "file_name": df["name"][rowId], #"Gen6/" + df["name"][rowId],
            "height": 224, #1456, 3000,
            "width": 224, #1088, 4000,
            "id": id
        },)
    cocoDict["annotations"].append(
        {
            "id": id + len(df.index),
            "image_id": id,
            "category_id": 5,
            "bbox": [df["x_left"][rowId], df["y_up"][rowId], df["x_right"][rowId] - df["x_left"][rowId], df["y_down"][rowId] - df["y_up"][rowId]],
            "segmentation": [[df["x_left"][rowId], df["y_up"][rowId], df["x_right"][rowId], df["y_up"][rowId], df["x_right"][rowId], df["y_down"][rowId], df["x_left"][rowId], df["y_down"][rowId]]],  # Because not supported only bbox
            "iscrowd": 0,
            "area": (df["x_right"][rowId] - df["x_left"][rowId]) * (df["y_down"][rowId] - df["y_up"][rowId])
        },
    )
    id += 1

# Convert and write JSON object to file
annFile = "edited-ann.json"
with open(os.path.join(folder, annFile), "w") as outfile: 
    json.dump(cocoDict, outfile)

"""
# Create tran/val split and save to coco folder tree
root_dir = "./GeneratedCoCo5" #"./Train_data"
ann_dir = os.path.join(root_dir, "annotations")
base_dirs = ["train", "val", "test"]

split_nums = [[0, math.ceil(len(df.index)*0)], [math.ceil(len(df.index)*0), math.ceil(len(df.index)*0)], [math.ceil(len(df.index)*0), len(df.index)]]
# Check sizes
#print(split_nums)
#print(len(cocoDict["images"][split_nums[0][0]:split_nums[0][1]]), len(cocoDict["images"][split_nums[1][0]:split_nums[1][1]]), len(cocoDict["images"][split_nums[2][0]:split_nums[2][1]]))
#print(len(df.index), 820+273+273)
#exit(0)
# Manage folders
if not os.path.exists(root_dir):
    os.mkdir(root_dir)
if not os.path.exists(ann_dir):
    os.mkdir(ann_dir)
for mode, s in zip(base_dirs, split_nums):
    mode_dir = os.path.join(root_dir, mode + "2017")
    #if not os.path.exists(mode_dir):
    #    os.mkdir(mode_dir)
    # Create ann 
    annFile = "{}_{}2017.json".format("instances", mode)
    annDict = copy.deepcopy(cocoDict)
    annDict["images"] = annDict["images"][s[0]:s[1]]
    annDict["annotations"] = annDict["annotations"][s[0]:s[1]]
    with open(os.path.join(ann_dir, annFile), "w") as outfile: 
        json.dump(annDict, outfile)
    # Move images
    #for img in tqdm(annDict["images"], "Copying images for {} mode".format(mode)):
    #    if not os.path.exists(os.path.join(mode_dir, img["file_name"])):
    #        shutil.copy2(os.path.join(folder, img["file_name"]), os.path.join(mode_dir, img["file_name"]))

            
# Check ann with pycocotools
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np

#coco = COCO(os.path.join(ann_dir, annFile))
coco = COCO(os.path.join(folder, annFile))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['detector'])
imgIds = coco.getImgIds(catIds=catIds)
#imgIds = coco.getImgIds(imgIds=[0])
ranIdx = np.random.randint(0,len(imgIds))
imgs = coco.loadImgs(imgIds[ranIdx:ranIdx+2])

for img in imgs:
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    I = io.imread(folder[:-5] + img['file_name'])
    plt.axis('off')
    plt.imshow(I)
    coco.showAnns(anns, draw_bbox=False)
    plt.show()
"""