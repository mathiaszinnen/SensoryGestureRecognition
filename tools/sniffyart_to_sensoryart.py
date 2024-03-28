import json
from statistics import class_dist, gesture_dist
from copy import deepcopy
import random

def separate_cats(gesture_string):
    if gesture_string in ['person', 'None'] :
        return []
    else:
        return [c.strip() for c in gesture_string.split(',')]
    
def create_gesture_string(catname, filter_cats):
    if catname == 'person':
        return 'None'
    gest_string = catname
    for cat in filter_cats:
        if cat in gest_string:
            gest_string = gest_string.replace(cat, '').strip().strip(',').strip()
            print('check')
    return gest_string

def get_filter_cats(coco, thresh=10):
    ## simple solution
    return ['vomiting', 'kissing']
    ## complicated solution
    #cd = class_dist(coco)
    #return [name for name, count in cd.items() if count < thresh]

def sniffyart_filter_cats(coco):
    # return categories to remove to obtain only categories present in sniffyart
    return ['playing music', 'praying', 'dancing', 'eating', 'reading', 'textile work', 'painting', 'sleeping', 'peeing', 'writing', 'vomiting', 'kissing']


def to_sniffyart_format(coco, humanart_template, filter_cats):
    coco = deepcopy(coco)
    catnames = {cat['id']: cat['name'] for cat in coco['categories']}
    filtered_anns = []
    for ann in coco['annotations']:
        catname = catnames[ann['category_id']]
        if catname in filter_cats:
            catname = 'person'
        ann['gesture'] = create_gesture_string(catname, filter_cats)
        ann['gestures'] = separate_cats(ann['gesture'])
        ann['category_id'] = 1
    

    categories = [
        {"id": 1,
         "name": "person", 
         "keypoints": humanart_template['categories'][0]['keypoints'],
         "skeleton": humanart_template['categories'][0]['skeleton']
        }
    ]
    coco['categories'] = categories

    return coco

def transfer(from_coco, to_coco, img_id):
    to_coco['images'].extend([img for img in from_coco['images'] if img_id == img['id']])
    to_coco['annotations'].extend([ann for ann in from_coco['annotations'] if ann['image_id'] == img_id])

    from_coco['images'] = [img for img in from_coco['images'] if img_id != img['id']]
    from_coco['annotations'] = [ann for ann in from_coco['annotations'] if img_id != ann['image_id']]
    return from_coco, to_coco

def increase_samples(gesture, src_coco, to_coco, tgt_number):
    n_samples = len([ann for ann in to_coco['annotations'] if gesture in ann['gestures']])
    print(f'Increasing number of {gesture} annotations from {n_samples} to {tgt_number}.')
    matching_anns_src = [ann for ann in src_coco['annotations'] if gesture in ann['gestures']]
    matching_imgs_src = list(set([ann['image_id'] for ann in matching_anns_src]))
    print(f'{len(matching_anns_src)} matching samples on {len(matching_imgs_src)} found.')

    while(n_samples < tgt_number):
        transfer_img_id = random.choice(matching_imgs_src)
        src_coco, to_coco = transfer(src_coco, to_coco, transfer_img_id)
        n_samples = len([ann for ann in to_coco['annotations'] if gesture in ann['gestures']])
        print(f'New number of samples: {n_samples}..')

    return src_coco, to_coco 

def generate_splits(coco, valid_txt, test_txt):
    with open(valid_txt) as f:
        valid_fns = [n.strip() for n in f.readlines()]
    with open(test_txt) as f:
        test_fns = [n.strip() for n in f.readlines()]

    valid_ids = [img['id'] for img in coco['images'] if img['file_name'] in valid_fns]
    test_ids = [img['id'] for img in coco['images'] if img['file_name'] in test_fns]
    train_ids = [img['id'] for img in coco['images'] if img['id'] not in valid_ids + test_ids]

    splits = []
    for split_ids in train_ids, valid_ids, test_ids:
        splits.append({
            'images': [img for img in coco['images'] if img['id'] in split_ids],
            'annotations': [ann for ann in coco['annotations'] if ann['image_id'] in split_ids],
            'categories': coco['categories']
        })
    #train, valid, test = splits

    cats_to_increase = min_samples()
    for name, n in cats_to_increase:
        for cur_split in splits[1], splits[2]:
            increase_samples(name, splits[0], cur_split, n)

    return splits


def min_samples():
    return [
        ('peeing',3),
        ('reading', 3),
        ('writing',3),
        ('sleeping',3),
        ('textile work', 3),
        ('painting', 5),
        ('playing music', 8)
    ]


def generate_split_txt(coco, txt_pth):
    with open(txt_pth, 'w') as f:
        for img in coco['images']:
            f.write(f'{img["file_name"]}\n')

def write_split_txt():
    with open('annotations/v2/train.json') as f:
        generate_split_txt(json.load(f), 'annotations/v2/train_images.txt')
    with open('annotations/v2/test.json') as f:
        generate_split_txt(json.load(f), 'annotations/v2/test_images.txt')
    with open('annotations/v2/valid.json') as f:
        generate_split_txt(json.load(f), 'annotations/v2/valid_images.txt')

def sanitize_ids(coco):
    imgmap = {}
    for i, img in enumerate(coco['images'], 1):
        imgmap[img['id']] = i
        img['id'] = i
    for i, ann in enumerate(coco['annotations'], 1):
        ann['id'] = i
        ann['image_id'] = imgmap[ann['image_id']]
    

def is_invalid(bbox):
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0:
        return True


def count_kpts(ann):
    kpts = ann['keypoints']
    n_kpts = 0
    for i in range(0,len(kpts), 3):
        if kpts[i+2] != 0:
            n_kpts+=1
    return n_kpts



def fix_errors(coco):
    #failing = [ann for ann in coco['annotations'] if ann['id'] in [451,452,453, 565,566,567,580,581,582]]
    failing = coco['annotations']
    sortouts = []
    for ann in failing:
        if is_invalid(ann['bbox']):
            sortouts.append(ann['id'])
        ann['num_keypoints'] = count_kpts(ann)
    print(f'Removing {len(sortouts)} invalid annotations..')
    coco['annotations'] = [ann for ann in coco['annotations'] if ann['id'] not in sortouts]

    # remove images without annotations
    ann_imgs = set([ann['image_id'] for ann in coco['annotations']])
    n_images_before = len(coco['images'])
    coco['images'] = [img for img in coco['images'] if img['id'] in ann_imgs]
    n_images_after = len(coco['images'])
    print(f'{n_images_before - n_images_after} images without annotations removed.')

    sanitize_ids(coco)
    return coco
    #return coco


def merge_splits(spls):
    base = spls[0]

    for spl in spls[1:]:
       base['annotations'] = base['annotations'] + spl['annotations']
       base['images'] = base['images'] + spl['images']

    fix_errors(spl)
    return base


def main():
    write_split_txt()
    # incorporate v1 images
    with open('annotations/v1/annotations_complete.json') as f:
        v1_complete = json.load(f)
    
    with open('annotations/human_art/humanart_filtered_coco.json') as f:
        human_art_coco = json.load(f)

    # read raw new images
    with open('annotations/cvat/manual_corrected.json') as f:
        updated_complete = json.load(f)

    filter_cats = get_filter_cats(updated_complete, thresh=10)
    sensory_art = to_sniffyart_format(updated_complete, human_art_coco, filter_cats)

    sniffyart_extended = to_sniffyart_format(updated_complete, human_art_coco, sniffyart_filter_cats(updated_complete))

    with open('annotations/v2/annotations_complete.json', 'w') as f:
        json.dump(sensory_art, f)
    
    with open('annotations/v2/sniffcats_only.json', 'w') as f:
        json.dump(sniffyart_extended, f)
   
   # stratified split of remaining images
    train_coco, valid_coco, test_coco = generate_splits(sensory_art, 'annotations/v1/valid_images.txt', 'annotations/v1/test_images.txt')

    with open('annotations/v2/train.json', 'w') as f:
        json.dump(train_coco, f)
    with open('annotations/v2/valid.json', 'w') as f:
        json.dump(valid_coco, f)
    with open('annotations/v2/test.json', 'w') as f:
        json.dump(test_coco, f)




if __name__ == '__main__':
    main()