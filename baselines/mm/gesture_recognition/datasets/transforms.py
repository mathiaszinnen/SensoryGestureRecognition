from typing import Optional, Dict

from mmdet.datasets.transforms import LoadAnnotations, Resize
from mmdet.registry import TRANSFORMS 

from mmpose.registry import TRANSFORMS as POSE_TRANSFORMS
from mmpose.datasets.transforms import BottomupGetHeatmapMask

from mmcv.transforms import BaseTransform
from mmcv.image import imflip_, imresize

import cv2
import numpy as np


@TRANSFORMS.register_module()
class LoadMultiTaskAnnotations(LoadAnnotations):
    """Load multiple types of annotations, including pose estimation keypoints."""
    def __init__(self,
                with_kpts=False,
                **kwargs):
        super().__init__(**kwargs)
        self.with_kpts = with_kpts


    def _load_kpts(self, results):
        gt_kpts = []
        kpts_vis = []
        for instance in results.get('instances', []):
            kps = instance['keypoints']
            xs = kps[0::3]
            ys = kps[1::3]
            vs = kps[2::3]

            gt_kpts.extend(np.dstack((xs,ys)))
            kpts_vis.append(vs)
        
        results['keypoints'] = np.array(gt_kpts, dtype=np.float32)
        results['keypoints_visible'] = np.array(kpts_vis)
        return results

    def transform(self, results):
        results = super().transform(results)
        if self.with_kpts:
            results = self._load_kpts(results)

        return results


@TRANSFORMS.register_module()
class MultiTaskResize(Resize):
    def _scale_kpts(self,results):
        if len(results.get('gt_keypoints', [])) > 0:
            results['gt_keypoints'] *= results['scale_factor']
        # todo: add clip border stuff here too?
        return results

    def transform(self, results: dict) -> dict:
        results =  super().transform(results)
        results = self._scale_kpts(results)

        results['input_size'] = self.scale
        return results


@TRANSFORMS.register_module()
class PackMultiTaskInputs(BaseTransform):
    def __init__(self, posepacker, detpacker):
        self.detpacker = TRANSFORMS.build(detpacker)
        self.posepacker = POSE_TRANSFORMS.build(posepacker)

    def transform(self, results):
        packed_pose = self.posepacker(results)
        packed_det = self.detpacker(results)


        return {
            'inputs': packed_det['inputs'],
            'det_samples': packed_det['data_samples'],
            'pose_samples': packed_pose['data_samples']
            # 'data_samples': {
            #     'det_samples': packed_det['data_samples'],
            #     'pose_samples': packed_pose['data_samples']
            # }
        }


@TRANSFORMS.register_module()
class FixedBottomupGetHeatmapMask(BottomupGetHeatmapMask):
    def transform(self, results: Dict) -> Optional[dict]:
        """Same as super().transform except it also works without warp stuff.
        """

        invalid_segs = results.get('invalid_segs', [])
        img_shape = results['img_shape']  # (img_h, img_w)
        input_size = results['input_size']
        mask = self._segs_to_mask(invalid_segs, img_shape)

        if not self.get_invalid:
            # Calculate the mask of the valid region by negating the
            # segmentation mask of invalid objects
            mask = np.logical_not(mask).astype(np.float32)

        # Apply an affine transform to the mask if the image has been
        # transformed
        if 'warp_mat' in results:
            warp_mat = results['warp_mat']

            mask = cv2.warpAffine(
                mask, warp_mat, input_size, flags=cv2.INTER_LINEAR)

        # Flip the mask if the image has been flipped
        if results.get('flip', False):
            flip_dir = results['flip_direction']
            if flip_dir is not None:
                mask = imflip_(mask, flip_dir)

        # Resize the mask to the same size of heatmaps
        if 'heatmaps' in results:
            heatmaps = results['heatmaps']
            if isinstance(heatmaps, list):
                # Multi-level heatmaps
                heatmap_mask = []
                for hm in results['heatmaps']:
                    h, w = hm.shape[1:3]
                    _mask = imresize(
                        mask, size=(w, h), interpolation='bilinear')
                    heatmap_mask.append(_mask)
            else:
                h, w = heatmaps.shape[1:3]
                heatmap_mask = imresize(
                    mask, size=(w, h), interpolation='bilinear')
        else:
            heatmap_mask = mask

        # Binarize the mask(s)
        if isinstance(heatmap_mask, list):
            results['heatmap_mask'] = [hm > 0.5 for hm in heatmap_mask]
        else:
            results['heatmap_mask'] = heatmap_mask > 0.5

        return results
