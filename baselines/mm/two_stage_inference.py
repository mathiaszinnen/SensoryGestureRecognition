from mmdet.apis import DetInferencer
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

from tqdm import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Perform inference via combining \
                                     a detection and classification models')
    parser.add_argument('--detector', help='config for the detection model to use',
                        default='baselines/mm/configs/person_detection.py')
    parser.add_argument('--detection_weights', help='path to the weights for the detection model',
                        default='/hdd/models/best_coco_bbox_mAP_epoch_166.pth')
    parser.add_argument('--classifier', help='config for the classification model to use',
                        default='baselines/mm/configs/crop_cls.py')
    parser.add_argument('--cls_weights', help='weights for the classification model',
                        default='/hdd/models/swinb_cls_best.pth')
    parser.add_argument('--det_thresh', type=float, help='keep only detection with confidence over det_thresh.',
                        default=.5)
    return parser.parse_args()

def handle_vis(cfg):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        visualization_hook['draw'] = True
        visualization_hook['test_out_dir'] = f'{cfg.work_dir}/twostage_debug_preds'
    return cfg


def parse_det_cfg(args, vis=False):
    det_cfg = Config.fromfile(args.detector)
    workdir = './work_dirs/twostage'
    det_cfg.work_dir=workdir
    if vis:
        det_cfg = handle_vis(det_cfg)
    det_cfg.test_evaluator.format_only=True
    det_cfg.test_evaluator.outfile_prefix=f'{workdir}/detections'
    det_cfg.load_from=args.detection_weights
    if vis:
        det_cfg.default_hooks.visualization.test_out_dir=f'{workdir}/detection'

    return det_cfg


def generate_person_boxes(det_cfg):
    det_runner = RUNNERS.build(det_cfg)
    det_runner.test()

    with open(f'{det_cfg.work_dir}/detections.bbox.json') as f:
        person_boxes = json.load(f)
    with open(f'{det_cfg.test_evaluator.ann_file}') as f:
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

    out_path_base = '/hdd/sniffyart/predictions'
    out_path = f'{out_path_base}/t{int(det_thresh*100)}_preds.json'
    with open(out_path, 'w') as f:
        json.dump(updated_anns, f)
    print(f'Predictions dumped to {out_path}')
            

    
if __name__ == '__main__':
    register_all_modules()
    main()