# https://www.kaggle.com/pestipeti/competition-metric-map-0-4/comments
import os, sys
import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
import json
from collections import defaultdict

"""
modify to dictionary format
"""


class MyCOCOeval(COCOeval):
    # def __init__(self, cocoGt=None, cocoDt=None, iouType="segm"):
    #     super().__init__(self, cocoGt=None, cocoDt=None, iouType="segm")

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]

            if len(s[s > -1]) == 0:
                aps = []
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])

                # cacluate AP(average precision) for each category
                aps = []
                num_classes = s.shape[2]  # 15
                avg_ap = 0.0
                if ap == 1:
                    for i in range(0, num_classes):
                        i_ap = np.mean(s[:, :, i, :])
                        aps.append(i_ap)
                        # print("category : {0} : {1}".format(i, i_ap))
                        avg_ap += np.mean(s[:, :, i, :])
                    # print("(all categories) mAP : {}".format(avg_ap / num_classes))

            # print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s, aps

        def _summarizeDets():

            out = _summarize(1)
            return out

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets

        self.stats = summarize()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class VinBigDataEval:
    """Helper class for calculating the competition metric.
    From - with huge modifications
    Original Authors: Peter (https://kaggle.com/pestipeti)
    See:
        https://www.kaggle.com/pestipeti/competition-metric-map-0-4

    Returns: None

    """

    def __init__(self, true_dict, cat=None, verbosity=0):

        self.true_dict = true_dict

        self.image_ids = sorted(true_dict.keys())

        self.cat = cat
        self.annotations = {
            "type": "instances",
            "images": self.gen_images(),
            "categories": self.gen_categories(),
            "annotations": self.gen_annotations(),
        }

        self.predictions = {
            "images": self.annotations["images"].copy(),
            "categories": self.annotations["categories"].copy(),
            "annotations": None,
        }

        self.verbosity = verbosity

    def gen_categories(self):
        # print("Generating category data...")

        cats = [
            "Aortic enlargement",  ### 0 614
            "Atelectasis",  ### 1 37
            "Calcification",  ### 2 90
            "Cardiomegaly",  ### 3 460
            "Consolidation",  ### 4 71
            "ILD",  ### 5 77
            "Infiltration",  ### 6 123
            "Lung Opacity",  ### 7 264
            "Nodule/Mass",  ### 8 165
            "Other lesion",  ### 9 227
            "Pleural effusion",  ### 10 206
            "Pleural thickening",  ### 11 396
            "Pneumothorax",  ### 12 20
            "Pulmonary fibrosis",  ### 13 323
            "No finding",  ### 14 2121
        ]

        if self.cat is not None:
            cats = [cats[self.cat]]

        results = []

        for icat, cat in enumerate(cats):
            results.append(
                {
                    "id": icat,
                    "name": cat,
                    "supercategory": "none",
                }
            )

        return results

    def gen_images(self):
        # print("Generating image data...")
        results = []

        for idx, img_id in enumerate(self.image_ids):

            # Add image identification.
            results.append(
                {
                    "id": img_id,
                }
            )

        return results

    def gen_annotations(self):  # , target_cat=None):
        # print("Generating annotation data...")
        k = 0
        results = []

        for idx, img_id in enumerate(self.image_ids):
            img_info = self.true_dict[img_id]
            for bbox in img_info["bbox"]:
                # bbox: x_min, y_min, x_max, y_max, cat
                cat_id = bbox[4]
                if self.cat is not None:
                    if int(cat_id) != self.cat:
                        continue
                    else:
                        cat_id = 0

                x_min, y_min, x_max, y_max = bbox[:4]
                results.append(
                    {
                        "id": k,
                        "image_id": img_id,
                        "category_id": int(cat_id),
                        "bbox": np.array([x_min, y_min, x_max, y_max]),
                        "segmentation": [],
                        "ignore": 0,
                        "area": (x_max - x_min) * (y_max - y_min),
                        "iscrowd": 0,
                    }
                )

                k += 1

        return results

    def gen_predictions(self, pred_dict):  # target_cat=None):
        # print("Generating prediction data...")
        k = 0
        results = []

        for idx, img_id in enumerate(self.image_ids):
            img_info = pred_dict[img_id]
            for bbox in img_info["bbox"]:
                cat_id = bbox[4]
                if self.cat is not None:
                    if int(cat_id) != self.cat:
                        continue
                    else:
                        cat_id = 0

                x_min, y_min, x_max, y_max = bbox[:4]
                results.append(
                    {
                        "id": k,
                        "image_id": img_id,
                        "category_id": int(cat_id),
                        "bbox": np.array([x_min, y_min, x_max, y_max]),
                        "segmentation": [],
                        "ignore": 0,
                        "area": (x_max - x_min) * (y_max - y_min),
                        "iscrowd": 0,
                        "score": bbox[5],
                    }
                )

                k += 1

        return results

    def evaluate(self, n_imgs=-1):
        if self.predictions["annotations"] is None:
            raise ("Please do .gen_predicitons() first!")

        """Evaluating your results

        Arguments:
            pred_df: pd.DataFrame your predicted results in the
                     competition output format.

            n_imgs:  int Number of images use for calculating the
                     result.All of the images if `n_imgs` <= 0

        Returns:
            COCOEval object
        """

        # if pred_dict is not None:
        # TODO:  asd
        # self.predictions["annotations"] = self.gen_predictions(
        #     pred_dict, self.image_ids
        # )

        if self.verbosity == 0:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

        coco_ds = COCO()
        coco_ds.dataset = self.annotations
        coco_ds.createIndex()

        coco_dt = COCO()
        coco_dt.dataset = self.predictions
        coco_dt.createIndex()

        imgIds = sorted(coco_ds.getImgIds())

        if n_imgs > 0:
            imgIds = np.random.choice(imgIds, n_imgs)

        cocoEval = MyCOCOeval(coco_ds, coco_dt, "bbox")
        cocoEval.params.imgIds = imgIds
        cocoEval.params.useCats = True
        cocoEval.params.iouType = "bbox"
        cocoEval.params.iouThrs = np.array([0.4])

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        if self.verbosity == 0:
            sys.stdout.close()
            sys.stdout = self._original_stdout

        return cocoEval


"""
usage: 
df = pd.read_csv("../input/vinbigdata-chest-xray-abnormalities-detection/train.csv")
df.fillna(0, inplace=True)
df.loc[df["class_id"] == 14, ['x_max', 'y_max']] = 1.0
df.head()

df = df.groupby(by=['image_id', 'class_id']).first().reset_index()

vineval = VinBigDataEval(df)

pred_df = df[["image_id"]]
pred_df = pred_df.drop_duplicates()
pred_df["PredictionString"] = "14 1.0 0 0 1 1"
pred_df.reset_index(drop=True, inplace=True)

cocoEvalRes = vineval.evaluate(pred_df)


"""