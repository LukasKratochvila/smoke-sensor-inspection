# Check ann with pycocotools
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import os

folder = "DS_sorted/Generated" #"Datasets/NoveAnotovana/Dataset02_COCO" #"Datasets/NoveAnotovana/Dataset01_COCO"
imgFolder = folder + "/."
annFile = "allGeneratedVal.json" ##

coco = COCO(os.path.join(folder, annFile))
#coco = COCO("realData.json")

# get all images containing given categories, select one at random
#catIds = coco.getCatIds(catNms=['s']) #['Sensor5']
#catIds = [0,1,2,3,4]
catIds = [0]
imgIds = coco.getImgIds(catIds=catIds)
#imgIds = coco.getImgIds(imgIds=[0])
imgs = coco.loadImgs(imgIds)

for img in imgs:
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    #I = io.imread(os.path.join(imgFolder, "images", os.path.basename(img['file_name'].split('\\')[-1])))
    I = io.imread(os.path.join(imgFolder, img['file_name']))
    plt.axis('off')
    plt.imshow(I)
    coco.showAnns(anns, draw_bbox=False)
    plt.show()
