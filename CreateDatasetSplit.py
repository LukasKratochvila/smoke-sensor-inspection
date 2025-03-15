import os
import json
import math
import copy
import shutil
import random
from pycocotools.coco import COCO
from tqdm import tqdm

source = "allDataBalancedTest.json"# "allData.json" "synteticReal.json" #"realData.json"
coco = COCO(source)
count = len(coco.dataset["images"])

# Create tran/val split and save to coco folder tree
root_dir = "./Datasets/NoveAnotovana/ObecneCidlo/DatasetD" # A B C
sourceFolder = "./Datasets/NoveAnotovana"

base_dirs = ["train", "test", "val"]
target_dirs = [os.path.join(root_dir, mode) for mode in base_dirs]
print(target_dirs)

nums = [0.6, 0.8]#[0.6, 0.8], [0.909002195, 0.954502196]
split_nums = [[0, math.ceil(count*nums[0])], [math.ceil(count*nums[0]), math.ceil(count*nums[1])], [math.ceil(count*nums[1]), count]]
# Check sizes
print(split_nums, len(coco.dataset["images"][split_nums[0][0]:split_nums[0][1]]), len(coco.dataset["images"][split_nums[1][0]:split_nums[1][1]]), len(coco.dataset["images"][split_nums[2][0]:split_nums[2][1]]))
exit(0)
# Manage folders
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

# For all data need shuffle
#random.shuffle(coco.dataset["images"])

for target_dir, s in zip(target_dirs, split_nums):
    # Create dirs
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    imgDir = os.path.join(target_dir, "images")
    if not os.path.exists(imgDir):
        os.mkdir(imgDir)

    # Create ann 
    annDict = COCO()
    annDict.dataset["info"] = copy.deepcopy(coco.dataset["info"])
    mode = os.path.basename(target_dir)
    annDict.dataset["info"]["description"] += f"-mode-{mode}"
    annDict.dataset["categories"] = coco.dataset["categories"]
    annDict.info()
    
    annDict.dataset["images"] = coco.dataset["images"][s[0]:s[1]]
    #annDict.dataset["annotations"] = list()
    annDict.createIndex()
    #print(annDict.getImgIds())
    #print(coco.getAnnIds(annDict.getImgIds()))
    annDict.dataset["annotations"] = coco.loadAnns(coco.getAnnIds(annDict.getImgIds()))
    print("#images:",len(annDict.dataset["images"]))
    print("#annotations:",len(annDict.dataset["annotations"]))

    with open(os.path.join(target_dir, "result.json"), "w") as outfile: 
        json.dump(annDict.dataset, outfile)
    # Move images
    for img in tqdm(annDict.dataset["images"], "Copying images for {} mode".format(mode)):
        if not os.path.exists(os.path.join(imgDir, img["file_name"])):
            shutil.copy2(os.path.join(sourceFolder, img["file_name"]), os.path.join(imgDir, os.path.basename(img["file_name"])))
    del annDict