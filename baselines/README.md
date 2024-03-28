# Sensory Gesture Recognition Methods

This folder contains the code to train models using the SensoryArt dataset. 

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

## Ultralytics

The code to train and evaluate yolov8 detection models and the multi-task detection/pose estimation models can be found in the `ultralytics` subfolder.