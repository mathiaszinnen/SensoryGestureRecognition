import json
from tqdm import tqdm
import os
import copy


METAINFO = {
    'classes':
    ('cooking', 'dancing', 'drinking', 'eating', 'holding the nose',
    'peeing', 'playing music', 'praying', 'reading', 'sleeping', 
    'smoking', 'sniffing', 'textile work', 'writing', 'painting', 
    'none'),
    'palette': 
    [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30)]
}


def to_gestures_coco(coco_path, tf):
    with open(coco_path) as f:
        person_coco = json.load(f)
    

    cats = [{"id": i, "name": n} for i,n in enumerate(METAINFO['classes'],1)]
    catname_to_id = {cat['name']: cat['id'] for cat in cats} 

    anns_updated = []
    for ann in tqdm(person_coco['annotations']):
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

    with open(tf, 'w') as f:
        json.dump(person_coco, f)


if __name__ == '__main__':
    annotations_dir = 'data/annotations'
    os.makedirs(f'{annotations_dir}/gesture_detection', exist_ok=True)
    
    for in_json in ['train.json', 'test.json', 'valid.json']:
        print(f'Converting {in_json}...')
        in_pth = f'{annotations_dir}/{in_json}'
        out_pth = f'{annotations_dir}/gesture_detection/{in_json}'

        to_gestures_coco(in_pth, out_pth)