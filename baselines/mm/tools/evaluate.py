import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import numpy as np
from torcheval.metrics import AUC

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='Metric to compute', choices = ['coco', 'multilabel'])
    parser.add_argument('gt_coco', help='Path to ground truth json in coco format')
    parser.add_argument('detections', help='Path to predictions json in coco format, only annotations array needed')
    parser.add_argument('conf_thresh', help='Keep only predictions with confidence above', type=float, default=.5)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.type == 'coco':
        compute_coco(args)
    if args.type == 'multilabel':
        compute_cls_metric(args)


def compute_coco(args):
    coco_gt = COCO(args.gt_coco)
    with open(args.detections) as f:
        preds_json = json.load(f)
    preds_json = [ann for ann in preds_json if ann['score'] >= args.conf_thresh]
    coco_pred = coco_gt.loadRes(preds_json)
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def precision(tp, fp):
    if tp+fp == 0:
        return 0
    return tp / (tp + fp)

def recall(tp, fn):
    return tp / (tp + fn)

def f1(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    if p + r == 0:
        return 0
    else:
        return 2 * (p * r) / (p + r)

def instances_to_labels(instances):
    return set([inst['category_id'] for inst in instances])


def update_cls_metrics(img_metrics, cls_metrics):
    for cls_id in img_metrics:
        cls_metrics[cls_id] += 1
    return cls_metrics

def compute_cls_metric(args):
    with open(args.gt_coco) as f:
        gt_coco = json.load(f)
    with open(args.detections) as f:
        detections = json.load(f)

    cls_ids = [cat['id'] for cat in gt_coco['categories']]
    cls_tps = {cls_id: 0 for cls_id in cls_ids}
    cls_fps = {cls_id: 0 for cls_id in cls_ids}
    cls_fns = {cls_id: 0 for cls_id in cls_ids}
    auc = AUC()
    for img in gt_coco['images']:
        img_gts = [ann for ann in gt_coco['annotations'] if ann['image_id'] == img['id']]
        img_dts = [ann for ann in detections 
                   if ann['image_id'] == img['id'] 
                   and ann['score'] >= args.conf_thresh]
        predicted_classes = instances_to_labels(img_dts)
        annotated_classes = instances_to_labels(img_gts)
        img_tps = predicted_classes.intersection(annotated_classes)
        img_fps = predicted_classes.difference(annotated_classes)
        img_fns = annotated_classes.difference(predicted_classes)

        cls_tps = update_cls_metrics(img_tps, cls_tps)
        cls_fps = update_cls_metrics(img_fps, cls_fps)
        cls_fns = update_cls_metrics(img_fns, cls_fns)

    cls_prec = np.array([precision(cls_tps[cls_id], cls_fps[cls_id]) for cls_id in cls_ids])
    cls_rec = np.array([recall(cls_tps[cls_id], cls_fns[cls_id]) for cls_id in cls_ids])
    cls_f1 = np.array([f1(cls_tps[cls_id], cls_fps[cls_id], cls_fns[cls_id]) for cls_id in cls_ids])

    print(f'Prec. macro: {np.average(cls_prec)}')
    print(f'Rec. macro: {np.average(cls_rec)}')
    print(f'f1 macro: {np.average(cls_f1)}')

if __name__ == '__main__':
    main()