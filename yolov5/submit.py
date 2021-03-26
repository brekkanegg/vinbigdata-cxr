import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from glob import glob
import pickle


def yolo2voc(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]

    """
    bboxes = bboxes.copy().astype(
        float
    )  # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height

    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]

    return bboxes


def submit(opt):
    # DATA_DIR = opt.data_dir

    with open(opt.data_dir + "/png_1024l/test_meta_dict.pickle", "rb") as f:
        test_dict = pickle.load(f)

    image_ids = []
    PredictionStrings = []

    for file_path in tqdm(glob(f"{opt.data_dir}/{opt.label_dir}/*txt")):
        image_id = file_path.split("/")[-1].split(".")[0]
        w, h = test_dict[image_id]["dim1"], test_dict[image_id]["dim0"]
        # w, h = test_df.loc[test_df.image_id==image_id,['width', 'height']].values[0]
        f = open(file_path, "r")
        data = (
            np.array(f.read().replace("\n", " ").strip().split(" "))
            .astype(np.float32)
            .reshape(-1, 6)
        )
        data = data[:, [0, 5, 1, 2, 3, 4]]
        bboxes = list(
            np.round(
                np.concatenate(
                    (data[:, :2], np.round(yolo2voc(h, w, data[:, 2:]))), axis=1
                ).reshape(-1),
                1,
            ).astype(str)
        )
        for idx in range(len(bboxes)):
            bboxes[idx] = str(int(float(bboxes[idx]))) if idx % 6 != 1 else bboxes[idx]
        if str(bboxes[0]) != "15":
            image_ids.append(image_id)
            if str(bboxes[0]) != "14":
                PredictionStrings.append(" ".join(bboxes))
            else:
                PredictionStrings.append("14 1 0 0 1 1")

        pred_df = pd.DataFrame(
            {"image_id": image_ids, "PredictionString": PredictionStrings}
        )
        # FIXME:
        # sub_df = pd.merge(test_df, pred_df, on = 'image_id', how = 'left').fillna("14 1 0 0 1 1")
        sub_df = pred_df.fillna("14 1 0 0 1 1")
        sub_df = sub_df[["image_id", "PredictionString"]]
        sub_df.to_csv(opt.data_dir + "/submission.csv", index=False)
        sub_df.tail()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/data/minki/kaggle/vinbigdata-cxr"
    )
    parser.add_argument(
        "--label_dir", type=str, default="yolov5/runs/detect/exp/labels"
    )
    parser.add_argument("--submit_dir", type=str, default="submission")

    opt = parser.parse_args()
    print(opt)

    submit(opt)
