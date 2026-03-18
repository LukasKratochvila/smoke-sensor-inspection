import copy
import io
from contextlib import redirect_stdout

import numpy as np
import pycocotools.mask as mask_util
import torch
import utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from faster_coco_eval.extra import PreviewResults, Curves


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead  got {type(iou_types)}")
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

        self.results = []

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

            if results:
                self.results.append(results[0])

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()
        return self.coco_eval["bbox"].stats[0]

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results
    
    # https://github.com/cocodataset/cocoapi/issues/381
    # https://github.com/facebookresearch/maskrcnn-benchmark/issues/94
    def plot_pr_curve(self, modelName, dataRoot, datasetMode, save=True):
        for iou_type in self.iou_types:    
            all_precision = self.coco_eval[iou_type].eval['precision']

            pr_5 = all_precision[0, :, 0, 0, 2] # data for IoU@0.5
            pr_7 = all_precision[4, :, 0, 0, 2] # data for IoU@0.7
            pr_9 = all_precision[8, :, 0, 0, 2] # data for IoU@0.9

            x = np.arange(0, 1.01, 0.01)
            plt.plot(x, pr_5, label='IoU@0.5')
            plt.plot(x, pr_7, label='IoU@0.7')
            plt.plot(x, pr_9, label='IoU@0.9')
            plt.legend(loc="upper right")
            plt.ylabel("Precision [-]")
            plt.xlabel("Recall [-]")
            plt.title("PR curve")

            if save:
                plt.savefig("-".join(["PRcurve","model", modelName.replace("/", "-"), "dataset", dataRoot.replace("/", "-"), datasetMode]) + ".png")
                plt.close()
            else:
                plt.show()

    # https://github.com/ultralytics/yolov5/issues/2782
    def plot_roc_curve(self, modelName, dataRoot, datasetMode, save=True):
        for iou_type in self.iou_types:
            imgCount = len(self.coco_eval[iou_type].evalImgs)
            gt_5 = np.array([np.all(self.coco_eval[iou_type].evalImgs[i]['gtMatches'][0,0]) for i in range(imgCount)]) # data for IoU@0.5
            gt_7 = np.array([np.all(self.coco_eval[iou_type].evalImgs[i]['gtMatches'][4,0]) for i in range(imgCount)]) # data for IoU@0.7
            gt_9 = np.array([np.all(self.coco_eval[iou_type].evalImgs[i]['gtMatches'][8,0]) for i in range(imgCount)]) # data for IoU@0.9
            
            dtScores = np.array([self.coco_eval[iou_type].evalImgs[i]['dtScores'][np.argsort(-np.array(self.coco_eval[iou_type].evalImgs[i]['dtScores']), kind='mergesort')[0]] for i in range(imgCount)])

            #print(imgCount, gtMatches, dtScores)
            fpr_5, tpr_5, _ = roc_curve(gt_5, dtScores, pos_label=1)
            roc_auc_5 = auc(fpr_5, tpr_5)
            fpr_7, tpr_7, _ = roc_curve(gt_7, dtScores, pos_label=1)
            roc_auc_7 = auc(fpr_7, tpr_7)
            fpr_9, tpr_9, _ = roc_curve(gt_9, dtScores, pos_label=1)
            roc_auc_9 = auc(fpr_9, tpr_9)


            x = np.arange(0, 1.01, 0.01)
            plt.xlim((-0.01, 1.01))
            plt.ylim((-0.01, 1.01))
            plt.plot(fpr_5, tpr_5, label=f'IoU@0.5 (AUC = {roc_auc_5})')
            plt.plot(fpr_7, tpr_7, label=f'IoU@0.7 (AUC = {roc_auc_7})')
            plt.plot(fpr_9, tpr_9, label=f'IoU@0.9 (AUC = {roc_auc_9})')
            plt.plot(x, x, label="Chance level (AUC = 0.5)", color="k", linestyle="--")
            plt.legend(loc="lower right")
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            plt.title("ROC curve")

            if save:
                plt.savefig("-".join(["ROCcurve","model", modelName.replace("/", "-"), "dataset", dataRoot.replace("/", "-"), datasetMode]) + ".png")
                plt.close()
            else:
                plt.show()

    def print_eval(self):
        for iou_type in self.iou_types:    
            print(self.coco_eval[iou_type].eval)

    def print_cm(self):
        coco_gt = copy.deepcopy(self.coco_gt)
        with redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(self.results) if self.results else COCO()
        results = PreviewResults(coco_gt, coco_dt, iou_tresh=0.5, iouType="bbox", useCats=False)
        results.display_matrix()

    def print_f1_confidence(self):
        coco_gt = copy.deepcopy(self.coco_gt)
        with redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(self.results) if self.results else COCO()
        cur = Curves(coco_gt, coco_gt, iou_tresh=0.5, iouType="bbox")
        cur.plot_pre_rec()
        cur.plot_f1_confidence()

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))
