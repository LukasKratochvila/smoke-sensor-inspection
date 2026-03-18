import os
import json
from pycocotools.coco import COCO

#annFiles = ["DS_sorted/Generated/BOSCH", "DS_sorted/Generated/ESSER", "DS_sorted/Generated/Real"] #Dataset_A 
#annFiles = ["Datasets/NoveAnotovana/Dataset03_COCO_fromGen6", "Datasets/NoveAnotovana/Dataset04_COCO_fromGen5", "Datasets/NoveAnotovana/Dataset01_COCO", "Datasets/NoveAnotovana/Dataset02_COCO"] #Dataset_B
#annFiles = ["Datasets/NoveAnotovana/Dataset01_COCO", "Datasets/NoveAnotovana/Dataset02_COCO"]#Dataset_C
annFiles = ["DS_sorted/Generated/BOSCH", "DS_sorted/Generated/ESSER", "DS_sorted/Generated/Real", "DS_sorted/Real"]

cocoMerged = COCO()
cocoMerged.dataset["info"] = {"description": "All generated data",
                              "url": "",
                              "version": "1.0",
                              "year": 2025,
                              "contributor": "BUT staff",
                              "date_created": "2025/1/15"}
cocoMerged.dataset["categories"] = [{"id": 1,
                              "name": "smoke_detector",
                              "supercategory": "Defect"}]
cocoMerged.dataset["annotations"] = list()
cocoMerged.dataset["images"] = list()

annIdShift = 0
imgIdShift = 0
for annFile in annFiles:
    coco = COCO(os.path.join(annFile, "train.json"))
    for value in coco.dataset["annotations"]:
        value["id"] += annIdShift
        value["image_id"] += imgIdShift
        value["category_id"] = 1
        x, y, width, height = value["bbox"]
        value["segmentation"] = [[x, y, width+x, y, width+x, y+height, x, y+height]]
        cocoMerged.dataset["annotations"].append(value)

    for value in coco.dataset["images"]:
        value["id"] += imgIdShift
        value["file_name"] = os.path.join(os.path.basename(annFile),"train", "images", value["file_name"].split("\\")[-1]) if annFile != "DS_sorted/Real" else "../" + os.path.join(os.path.basename(annFile),"train", "images", value["file_name"].split("\\")[-1]) 
        cocoMerged.dataset["images"].append(value)
    
    annIdShift=coco.dataset["annotations"][-1]["id"] + 1
    imgIdShift=coco.dataset["images"][-1]["id"] + 1

cocoMerged.createIndex()

# Convert and write JSON object to file
with open(os.path.join("MixTrain.json"), "w") as outfile: #"allData.json" "synteticReal.json" "realData.json"
    json.dump(cocoMerged.dataset, outfile)
