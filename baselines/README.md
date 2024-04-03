# Sensory Gesture Recognition Methods

This folder contains the code to reproduce the results reported in \[1\] using the SensoryArt dataset. 
The dataset features person boxes, gesture labels, and pose estimation keypoints.

Below are instructions on how to train and evaluate models for each of these label types.


## Getting started
- Download and prepare the dataset using the `download_dataset.py` script in the `tools/` directory. The dataset will be downloaded to `data/` in the the project root folder.
- Install dependencies for the task you want to evaluate. For mmdetection, mmpretrain, or mmpose models follow the instructions from the respective openmmlab github pages. For the YOLO models just run `pip install ultralytics`.

## Person Detection
- The DINO SWIN_B baseline model can be trained by invoking `mm/train.py` with the `mm/configs/person_detection.py` config. Additional backbones, detection heads, and model configurations can be found in the `mm/configs/detection/` directory.
- Evaluation of the mmdetection models is done via the `mm/test.py <<config>> <<ckpt>>`. Replace `<<config>>` with the config of the model you want to evaluate, and `<<ckpt>>` with the path to the trained checkpoint. 
- YOLO models are trained using `ultralytics/train.py` and evaluated using `ultralytics/test.py`. To prepare the dataset in coco format, please run `tools/generate_yolo.py person` from the project root. 

## Gesture Detection
- To prepare the dataset for gesture detection run `tools/generate_multigesture.py`

## Human Pose Estimation

## Gesture Classification

## Multi-Task Approaches

## MM
Most models are trainied using the openmmlab frameworks MMDetection, MMPretrain, MMPose, or combinations of these. 
All openmmlab-based algorithms are to be found in the `mm` subfolder.
You can train those algorithms using `mm/train.py <<config>>`, passing the respective configs (see below). Evaluation can be done by invoking `mm/test.py <<config>> <<checkpoint>>`. Both commands assume that your working directory is `baselines`.  
All configs can be found in the `mm/configs/` folder.


| Config | Task |
| --- | --- |
| gesture_detection.py | Direct Gesture Detection  |
| person_detection.py | Person Detection |
| crop_cls.py | Gesture classification of croppedout persons | 

To evaluate subsequent detection and classification models, invoke `mm/two_stage_inference.py`.

## Ultralytics

The code to train and evaluate yolov8 detection models and the multi-task detection/pose estimation models can be found in the `ultralytics` subfolder.