# Tools for Sensory Gesture Recognition

## Dataset download
Use download_dataset.py to download and extract the dataset in the `data` folder of the root repository

## Dataset conversion
- **generate_multigesture.py** converts the annotations from the sensoryart format to default coco format with the gestures as classes.
- **generate_crop_cls.py** generates a dataset for multilabel gesture classification of the cropped persons based on the sensoryart person boxes.
- **generate_yolo.py** generates a dataset for yolo detection, specify type person or gesture depending on whether persons should be detected or directly gestures.
- **generate_cls.py** generates a dataset for multilabel image-level gesture classification TODO
- **sniffyart_to_sensoryart.py** defines multiple functions used to extend the sniffyart dataset and generate sensoryart from it, including split generation. This script will probably not be needed again but is still included to increase the transparency and reproducibility of the dataset creation process.