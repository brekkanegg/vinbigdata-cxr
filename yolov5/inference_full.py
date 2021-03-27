import os

project_name = input("project_name: ")
gpu = input("Gpu: ")

# 1. detect
os.system(
    f"python detect.py --img 1024 --weights /data/minki/kaggle/vinbigdata-cxr/yolov5/runs/train/{project_name}/weights/best.pt --device {gpu} --name {project_name} --iou-thres 0.4 --conf-thres 0.007 --save-txt --save-conf"
)

# 2. apply post process
os.system(
    f"python apply_postprocess.py --src /data/minki/kaggle/vinbigdata-cxr/yolov5/runs/detect/{project_name}/labels --dst /data/minki/kaggle/vinbigdata-cxr/yolov5/runs/detect/{project_name}/labels_post"
)

# 3. make submit file
os.system(
    f"python submit.py --label_dir /data/minki/kaggle/vinbigdata-cxr/yolov5/runs/detect/{project_name}/labels_post --submit_name {project_name}_post"
)
