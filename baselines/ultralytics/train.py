from ultralytics import YOLO
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='Type of model to train', choices=['persons', 'gestures', 'multitask'])
    return parser.parse_args()


def main():
    args = parse_args() 

    data_yaml = f'datasets/yolo_{args.type}/sensoryart.yaml'

    if args.type in ['persons', 'gestures']:
        model = YOLO('yolov8x.pt')
    else:
        model = YOLO('yolov8x-pose.yaml')

    results = model.train(data=data_yaml, epochs=200, imgsz=384)



if __name__ == '__main__':
    main()