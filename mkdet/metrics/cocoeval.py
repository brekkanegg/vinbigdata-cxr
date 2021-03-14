# https://www.kaggle.com/pestipeti/competition-metric-map-0-4/comments
import os, sys
import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

"""
modify to dictionary format
"""


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class VinBigDataEval:
    """Helper class for calculating the competition metric.

    You should remove the duplicated annoatations from the `true_df` dataframe
    before using this script. Otherwise it may give incorrect results.

        >>> vineval = VinBigDataEval(valid_df)
        >>> cocoEvalResults = vineval.evaluate(pred_df)

    Arguments:
        true_df: pd.DataFrame Clean (no duplication) Training/Validating dataframe.

    Authors:
        Peter (https://kaggle.com/pestipeti)

    See:
        https://www.kaggle.com/pestipeti/competition-metric-map-0-4

    Returns: None

    """

    def __init__(self, true_dict):

        self.true_dict = true_dict

        # self.image_ids = true_df["image_id"].unique()
        self.image_ids = sorted(true_dict.keys())
        self.annotations = {
            "type": "instances",
            "images": self.__gen_images(self.image_ids),
            "categories": self.__gen_categories(),
            "annotations": self.__gen_annotations(self.image_ids),
        }

        self.predictions = {
            "images": self.annotations["images"].copy(),
            "categories": self.annotations["categories"].copy(),
            "annotations": None,
        }

    def __gen_categories(self):
        # print("Generating category data...")

        cats = [
            "Aortic enlargement",  ###
            "Atelectasis",
            "Calcification",
            "Cardiomegaly",  ###
            "Consolidation",
            "ILD",
            "Infiltration",
            "Lung Opacity",
            "Nodule/Mass",
            "Other lesion",  ###
            "Pleural effusion",
            "Pleural thickening",
            "Pneumothorax",
            "Pulmonary fibrosis",
            "No finding",
        ]

        # results = []

        # if "class_name" not in df.columns:
        #     df["class_name"] = df["class_id"]

        # cats = df[["class_name", "class_id"]]
        # cats = cats.drop_duplicates().sort_values(by="class_id").values

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

    def __gen_images(self, image_ids):
        # print("Generating image data...")
        results = []

        for idx, image_id in enumerate(image_ids):

            # Add image identification.
            results.append(
                {
                    "id": idx,
                }
            )

        return results

    def __gen_annotations(self, image_ids):
        # print("Generating annotation data...")
        k = 0
        results = []

        for idx, img_id in enumerate(image_ids):
            img_info = self.true_dict[img_id]
            for bbox in img_info["bbox"]:
                # bbox: x_min, y_min, x_max, y_max, cat
                cat_id = bbox[4]
                x_min, y_min, x_max, y_max = bbox[:4]
                results.append(
                    {
                        "id": k,
                        "image_id": idx,
                        "category_id": int(cat_id),
                        "bbox": np.array([x_min, y_min, x_max, y_max]),
                        "segmentation": [],
                        "ignore": 0,
                        "area": (x_max - x_min) * (y_max - y_min),
                        "iscrowd": 0,
                    }
                )

                k += 1

        # for idx, image_id in enumerate(image_ids):
        #     # Add image annotations
        #     for i, row in df[df["image_id"] == image_id].iterrows():
        #         results.append(
        #             {
        #                 "id": k,
        #                 "image_id": idx,
        #                 "category_id": row["class_id"],
        #                 "bbox": np.array(
        #                     [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
        #                 ),
        #                 "segmentation": [],
        #                 "ignore": 0,
        #                 "area": (row["x_max"] - row["x_min"])
        #                 * (row["y_max"] - row["y_min"]),
        #                 "iscrowd": 0,
        #             }
        #         )

        #         k += 1

        return results

    # def __decode_prediction_string(self, pred_str):
    #     data = list(map(float, pred_str.split(" ")))
    #     data = np.array(data)

    #     return data.reshape(-1, 6)

    def __gen_predictions(self, pred_dict, image_ids):
        # print("Generating prediction data...")
        k = 0
        results = []

        for idx, img_id in enumerate(image_ids):
            img_info = pred_dict[img_id]
            for bbox in img_info["bbox"]:
                cat_id = bbox[4]
                x_min, y_min, x_max, y_max = bbox[:4]
                results.append(
                    {
                        "id": k,
                        "image_id": idx,
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

        # for i, row in df.iterrows():

        #     image_id = row["image_id"]
        #     preds = self.__decode_prediction_string(row["PredictionString"])

        #     for j, pred in enumerate(preds):

        #         results.append(
        #             {
        #                 "id": k,
        #                 "image_id": int(np.where(image_ids == image_id)[0]),
        #                 "category_id": int(pred[0]),
        #                 "bbox": np.array([pred[2], pred[3], pred[4], pred[5]]),
        #                 "segmentation": [],
        #                 "ignore": 0,
        #                 "area": (pred[4] - pred[2]) * (pred[5] - pred[3]),
        #                 "iscrowd": 0,
        #                 "score": pred[1],
        #             }
        #         )

        #         k += 1

        return results

    def evaluate(self, pred_dict, n_imgs=-1):
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
        self.predictions["annotations"] = self.__gen_predictions(
            pred_dict, self.image_ids
        )

        # with HiddenPrints():

        coco_ds = COCO()
        coco_ds.dataset = self.annotations
        coco_ds.createIndex()

        coco_dt = COCO()
        coco_dt.dataset = self.predictions
        coco_dt.createIndex()

        imgIds = sorted(coco_ds.getImgIds())

        if n_imgs > 0:
            imgIds = np.random.choice(imgIds, n_imgs)

        cocoEval = COCOeval(coco_ds, coco_dt, "bbox")
        cocoEval.params.imgIds = imgIds
        cocoEval.params.useCats = True
        cocoEval.params.iouType = "bbox"
        cocoEval.params.iouThrs = np.array([0.4])

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

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