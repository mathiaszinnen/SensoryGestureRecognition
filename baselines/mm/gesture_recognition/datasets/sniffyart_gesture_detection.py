from typing import List
from mmdet.datasets.coco import CocoDataset
from mmdet.registry import DATASETS
from mmengine.fileio import get_local_path
import json
import copy
import os
import torch


@DATASETS.register_module()
class SniffyArtGestureDetection(CocoDataset):
    METAINFO = {
        'classes':
        ('cooking', 'dancing', 'drinking', 'eating', 'holding the nose',
        'peeing', 'playing music', 'praying', 'reading', 'sleeping', 
        'smoking', 'sniffing', 'textile work', 'writing', 'painting', 
        'none'),
        'palette': 
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (255,255,0), (0,255,0)]
    }
        
    def _to_gestures_coco(self, coco_path,tempfile):
        with open(coco_path) as f:
            person_coco = json.load(f)
        

        cats = [{"id": i, "name": n} for i,n in enumerate(SniffyArtGestureDetection.METAINFO['classes'],1)]
        catname_to_id = {cat['name']: cat['id'] for cat in cats} 

        anns_updated = []
        for ann in person_coco['annotations']:
            if len(ann['gestures']) == 0:
                updated_ann = copy.deepcopy(ann)
                updated_ann['category_id'] = catname_to_id['none']
                updated_ann['id'] = len(anns_updated) + 1
                anns_updated.append(updated_ann)
            else:
                for gesture in ann['gestures']:
                    updated_ann = copy.deepcopy(ann)
                    updated_ann['category_id'] = catname_to_id[gesture]
                    updated_ann['id'] = len(anns_updated) + 1
                    anns_updated.append(updated_ann)

        person_coco['categories'] = cats
        person_coco['annotations'] = anns_updated

        with open(tempfile, 'w') as f:
            json.dump(person_coco,f)
    

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            try:
                local_rank = os.environ['LOCAL_RANK']
            except KeyError:
                local_rank = 0
            tempfile = f'tmpdir/{local_rank}/{os.path.basename(local_path)}'
            self._to_gestures_coco(local_path, tempfile)
            self.coco = self.COCOAPI(tempfile)
        try:
            torch.distributed.barrier()
        except (RuntimeError, ValueError):
            # single gpu run -> just go on
            pass
        
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list