from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
import yaml
import os
import copy
import argparse


def val_on_test(model, yaml_pth):
    print('val on test')
    with open(yaml_pth) as f:
        ds_yaml = yaml.safe_load(f)
    backup_path = f'{os.path.splitext(yaml_pth)[0]}_backup.yaml'
    with open(backup_path, 'w') as f: # save old yaml to disc in case execution is interrupted
        yaml.dump(ds_yaml, f)
    test_yaml = copy.deepcopy(ds_yaml)
    test_yaml['val'] = test_yaml['test']
    with open(yaml_pth, 'w') as f:
        yaml.dump(test_yaml, f)
    results = model.val()
    with open(yaml_pth, 'w') as f:
        yaml.dump(ds_yaml, f)
    os.remove(backup_path)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='type of model to train', choices=['persons', 'gestures', 'multitask', 'poses'])
    parser.add_argument('checkpoint', help='path to trained checkpoint')
    parser.add_argument('--val', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args() 

    data_yaml = f'datasets/yolo_{args.type}/sensoryart.yaml'

    model = YOLO(args.checkpoint)

    if args.val == True:
        model.val()
        print('yay')
    else:
        val_on_test(model, data_yaml)
        print('yay')



if __name__ == '__main__':
    main()