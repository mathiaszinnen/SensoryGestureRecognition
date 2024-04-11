# adopted from https://github.com/ultralytics/JSON2YOLO/blob/master/general_json2yolo.py
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import shutil
import argparse
import yaml

def make_dirs(dir):
    """Creates a directory with subdirectories 'labels' and 'images', removing existing ones."""
    dir = Path(dir)
    if dir.exists():
        print("Dataset target directory already exists. Skipping.")
        exit()

    for p in dir, dir / "labels", dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir

def convert_coco_json(src_imgs, src_anns, out_dir, use_keypoints=False):
    """Converts SENSORYART JSON format to YOLO label format, with options for segments and class mapping."""
    json_dir = Path(src_anns)
    save_dir = make_dirs(out_dir)
    out_yaml = dict()
    out_yaml['path'] = out_dir.replace('datasets/','')

    # Import json
    for json_file in [Path(json_dir,split) for split in ['train.json', 'valid.json', 'test.json']]:
        fn = Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")  # folder name
        fn.mkdir()
        

        imgs_dir = Path(save_dir) / "images" / json_file.stem
        imgs_dir.mkdir()
        out_yaml[json_file.stem] = f'images/{json_file.stem}'

        with open(json_file) as f:
            data = json.load(f)

        # Create categories array for yaml
        out_yaml['names'] = {i: n for i,n in enumerate([cat['name'] for cat in data['categories']])}

        # Create image dict
        images = {"%g" % x["id"]: x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images["%g" % img_id]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            keypoints = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = ann["category_id"] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                if use_keypoints:
                    out_yaml['kpt_shape'] = [17, 3] 
                    out_yaml['flip_idx'] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
                    ann_kpts = np.array(ann['keypoints'], dtype=np.float64)
                    ann_kpts[0::3] /= w # normalize x
                    ann_kpts[1::3] /= h # normalize y
                    keypoints.append(ann_kpts)
                    

            # Copy images
            shutil.copyfile(f'{src_imgs}/{f}', f'{imgs_dir}/{f}'), 


            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*bboxes[i],)  # cls, box or segments
                    if use_keypoints:
                        line += (*keypoints[i],)
                    file.write(("%g " * len(line)).rstrip() % line + "\n")
    
    # Write yolo yaml
    out_yaml['val'] = out_yaml['valid']
    del out_yaml['valid']
    with open(f'{out_dir}/sensoryart.yaml', 'w') as f:
        yaml.dump(out_yaml, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, choices=['person', 'gesture','pose','multitask'])
    args = parser.parse_args()

    use_kpts = False
    src_imgs = 'data/images'
    if args.type == 'person':
        src_anns = 'data/annotations/'
        out_dir = 'datasets/yolo_persons'
    elif args.type == 'gesture':
        src_anns = 'data/annotations/gesture_detection'
        out_dir = 'datasets/yolo_gestures'
    elif args.type == 'pose':
        src_anns = 'data/annotations/'
        out_dir = 'datasets/yolo_poses'
        use_kpts=True
    elif args.type == 'multitask':
        src_anns = 'data/annotations/gesture_detection'
        out_dir = 'datasets/yolo_multitask'
        use_kpts=True


    convert_coco_json(src_imgs=src_imgs, src_anns=src_anns, out_dir=out_dir, use_keypoints=use_kpts)