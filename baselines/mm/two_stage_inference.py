from mmdet.apis import DetInferencer
import cv2
from mmengine.config import Config
from mmdet.registry import DATASETS, RUNNERS
from mmpretrain.registry import RUNNERS as CLS_RUNNERS, TRANSFORMS
from mmpretrain.datasets import remove_transform
from mmengine.dataset import Compose
from mmcv.image import imread, imwrite
from mmdet.engine.hooks.utils import trigger_visualization_hook
import argparse
import os
import torch
from typing import List
from mmpretrain.structures import DataSample
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmdet.utils import register_all_modules
import matplotlib.pyplot as plt
from mmdet.visualization.palette import _get_adaptive_scales
import random

from tqdm import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Perform inference via combining \
                                     a detection and classification models')
    parser.add_argument('--detector', help='config for the detection model to use',
                        default='baselines/mm/configs/person_detection.py')
    parser.add_argument('--detection_weights', help='path to the weights for the detection model',
                        default='/hdd/models/best_coco_bbox_mAP_epoch_166.pth')
                        # default='/hdd/models/dino_swinb_person_best.pth')
    parser.add_argument('--classifier', help='config for the classification model to use',
                        default='baselines/mm/configs/crop_cls.py')
    parser.add_argument('--cls_weights', help='weights for the classification model',
                        default='/hdd/models/swinb_cls_best.pth')
    parser.add_argument('--det_thresh', type=float, help='keep only detection with confidence over det_thresh.',
                        default=.5)
    parser.add_argument('--vis', help='Draw visualizations of classifications and detections', action='store_true')
    parser.add_argument('--val', help='Create predictions for the validation split', action='store_true')
    return parser.parse_args()

def handle_vis(cfg):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        visualization_hook['draw'] = True
        visualization_hook['test_out_dir'] = f'{cfg.work_dir}/twostage_debug_preds'
    return cfg


def parse_det_cfg(args):
    det_cfg = Config.fromfile(args.detector)
    workdir = './work_dirs/twostage'
    det_cfg.work_dir=workdir
    if args.vis:
        det_cfg = handle_vis(det_cfg)
    det_cfg.test_evaluator.format_only=True
    det_cfg.test_evaluator.outfile_prefix=f'{workdir}/detections'
    det_cfg.load_from=args.detection_weights
    if args.vis:
        det_cfg.default_hooks.visualization.test_out_dir=f'{workdir}/detection'
    if args.val:
        det_cfg.test_dataloader.dataset = det_cfg.val_dataloader.dataset
        det_cfg.test_evaluator.ann_file = det_cfg.val_evaluator.ann_file

    return det_cfg


def generate_person_boxes(det_cfg, val=False):
    ann_file = det_cfg.test_evaluator.ann_file

    det_runner = RUNNERS.build(det_cfg)
    det_runner.test()

    with open(f'{det_cfg.work_dir}/detections.bbox.json') as f:
        person_boxes = json.load(f)
    with open(ann_file) as f:
        coco = json.load(f)
    coco['annotations'] = person_boxes
    return coco


def parse_cls_cfg(args):
    cls_cfg = Config.fromfile(args.classifier)
    cls_cfg.load_from=args.cls_weights
    cls_cfg.work_dir = './work_dirs/twostage'

    return cls_cfg


def create_cls_pipeline(cls_cfg):
    inference_pipeline_cfg = cls_cfg.test_dataloader.dataset.pipeline
    inference_pipeline_cfg = remove_transform(inference_pipeline_cfg,
                                             'LoadImageFromFile')
    inference_pipeline = Compose(
        [TRANSFORMS.build(t) for t in inference_pipeline_cfg]
    )
    return inference_pipeline


def get_img_basedir(cfg):
    ds = cfg.val_dataloader.dataset
    return f'{ds.data_root}/{ds.data_prefix.img}'


def make_eval_ds(args):
    eval_cfg = Config.fromfile(args.eval_cfg)
    register_all_modules()
 
    try:
        eval_ds = DATASETS.build(eval_cfg.val_dataloader.dataset)
    except FileNotFoundError:
        os.makedirs('tmpdir/0', exist_ok=True)
        eval_ds = DATASETS.build(eval_cfg.val_dataloader.dataset)

    return eval_ds


def xywh_to_xyxy_sane(box, img):
    img_y, img_x = img.shape[:2]
    x,y,w,h = list(map(int,box))
    x = max(x, 0)
    y = max(y, 0)
    x2 = min(x+w, img_x)
    y2 = min(y+h, img_y)
    return x,y,x2,y2


def get_random_color():
    return (random.randint(0,255),random.randint(0,255),random.randint(0,255))


def draw_boxes(img, anns, coco_img, visualizer):
    visualizer.set_image(img)
    anns = [ann for ann in anns if ann['image_id'] == coco_img['id']]
    classes = visualizer.dataset_meta['classes']
    colors = [get_random_color() for _ in range(len(classes))]
    bboxes = []
    for ann in anns:
        x,y,w,h = list(map(int,ann['bbox']))
        bboxes.append([x,y,x+w,y+h])
    bboxes = torch.Tensor(bboxes)
    visualizer.draw_bboxes(
        bboxes=bboxes,
        edge_colors=colors
    )
    positions = bboxes[:,:2] + 2
    areas = (bboxes[:, 3] - bboxes[:, 1]) * (
    bboxes[:, 2] - bboxes[:, 0])
    scales = _get_adaptive_scales(areas)
    labels = [ann['category_id'] for ann in anns]
    scores = [ann['score'] for ann in anns]

    for i, (pos, label, score) in enumerate(zip(positions, labels, scores)):
        label_text = classes[label-1] 
        score = round(float(score) * 100, 1)
        label_text += f': {score}'

        newpos = torch.Tensor([pos[0], pos[1] + 2*i])

        visualizer.draw_texts(
            label_text,
            newpos,
            colors=colors[label-1],
            font_sizes=int(13 * scales[i]),
            bboxes=[{
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            }])

    imwrite(visualizer.get_image(), f'twostage_vis/{coco_img["file_name"]}')
    return


def main():
    args = parse_args() 

    det_cfg = parse_det_cfg(args)
    person_boxes = generate_person_boxes(det_cfg)

    cls_cfg = parse_cls_cfg(args) 
    
    cls_runner = CLS_RUNNERS.build(cls_cfg)
    cls_runner.test() # there must be a better way to trigger the model load the weights
    cls_model = cls_runner.model

    tfms = create_cls_pipeline(cls_cfg)


    det_thresh = args.det_thresh
    cls_thresh = .0
    img_basedir = get_img_basedir(det_cfg)

    updated_anns = []
    for coco_img in tqdm(person_boxes['images']):
        img = imread(f'{img_basedir}/{coco_img["file_name"]}')
        img_persons = [ann for ann in person_boxes['annotations'] 
                       if ann['image_id'] == coco_img['id'] 
                       and ann['score'] > det_thresh]
        for person in img_persons:
            x1,y1,x2,y2 = xywh_to_xyxy_sane(person['bbox'], img)
            if x2-x1 == 0 or y2-y1 == 0:
                print('invalid prediction box found')
                continue #ignore empty boxes
            crop = img[y1:y2, x1:x2] 
            cropdict = dict(
                img=crop,
                img_shape=crop.shape[:2],
                orig_shape=crop.shape[:2]
            )
            cropdict = tfms(cropdict)
            cropdict['inputs'] = cropdict['inputs'][None,:,:,:]
            cropdict['data_samples'] = [cropdict['data_samples']]
            results = cls_model.test_step(cropdict)[0]
            positives = torch.where(results.pred_score > cls_thresh)[0]
            for label in positives:
                score = results.pred_score[label]
                updated_anns.append({
                    'bbox': person['bbox'],
                    'image_id': coco_img['id'],
                    'category_id': label.detach().cpu().item()+1,
                    'score': score.detach().cpu().item()
                })
        if args.vis:
            box_anns = [ann for ann in updated_anns if ann['image_id'] == coco_img['id']]
            draw_boxes(img, box_anns, coco_img, cls_runner.visualizer)


    if args.val:
        out_path_base = '/hdd/sniffyart/predictions/val'
    else:
        out_path_base = '/hdd/sniffyart/predictions/test'

    out_path = f'{out_path_base}/t{int(det_thresh*100)}_preds.json'
    with open(out_path, 'w') as f:
        json.dump(updated_anns, f)
    print(f'Predictions dumped to {out_path}')
            

    
if __name__ == '__main__':
    register_all_modules()
    main()