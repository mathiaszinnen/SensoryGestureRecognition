import json
import glob
import os
import cv2
import json
import os
from tqdm import tqdm
import xml.etree.cElementTree as ET


def crop_box(img, ann, context_percentage=0):
    [x,y,w,h]= list(map(int,ann['bbox']))
    x_offset = round(.5 * w * context_percentage)
    y_offset = round(.5 * h * context_percentage)
    xmin = max(0, x - x_offset)
    xmax = min(img.shape[1], x + w + x_offset)
    ymin = max(0, y - y_offset)
    ymax = min(img.shape[0], y + h + y_offset)
    return img[ymin:ymax, xmin:xmax]

def extract_labels(ann):
    if len(ann['gestures']) == 0:
        return ['none']
    else:
        return ann['gestures']

def write_to_classtxt(labels,basedir,imname,split):
    imname = os.path.splitext(os.path.basename(imname))[0]
    for gest in labels:
        txt_pth = f'{basedir}/{gest}_{split}.txt'
        with open(txt_pth, 'a+') as f:
            f.write(f'{imname} -1\n')
    with open(f'{basedir}/{split}.txt', 'a+') as f:
        f.write(f'{imname}\n')

def create_xml(labels,basedir, imname, img):
    w,h,c = img.shape
    root = ET.Element("annotations")
    ET.SubElement(root, "folder").text = "VOC2012"
    ET.SubElement(root, "filename").text = imname
    size = ET.SubElement(root,"size")
    ET.SubElement(size,"width").text = str(w)
    ET.SubElement(size,"height").text = str(h)
    ET.SubElement(size,"depth").text = str(c)
    for gest in labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj,"name").text = gest
        ET.SubElement(obj,"difficult").text = "0"
    tree = ET.ElementTree(root)
    imname = os.path.splitext(imname)[0]
    tree.write(f'{basedir}/{imname}.xml')


def main():
    data_dir = 'data'
    coco_pths = [
        f'{data_dir}/annotations/train.json',
        f'{data_dir}/annotations/valid.json',
        f'{data_dir}/annotations/test.json',
    ]
    img_src_dir = f'{data_dir}/images/'
    tgt_base = f'{data_dir}/cls/VOC2012'
    img_tgt_dir = f'{tgt_base}/JPEGImages'
    ann_tgt_dir = f'{tgt_base}/ImageSets/Main'
    xml_tgt_dir = f'{tgt_base}/Annotations'
    for dir in [img_tgt_dir, ann_tgt_dir, xml_tgt_dir]:
        os.makedirs(dir, exist_ok=True)
    
    for spl, coco_pth in zip(['train', 'val', 'test'], coco_pths):
        print(f'Creating classification structure for {spl} split...')
        with open(coco_pth) as f:
            coco = json.load(f)
        catnames = {cat['id']: cat['name'] for cat in coco['categories']}
        for name in catnames.keys():
            os.makedirs(f'{data_dir}/cls/{spl}/{name}', exist_ok=True)
        for ann in tqdm(coco['annotations']):
            fn = [img['file_name'] for img in coco['images'] if ann['image_id'] == img['id']][0] 
            bn, ext = os.path.splitext(fn)
            crop_fn = f'{bn}_{ann["id"]}{ext}'
            img = cv2.imread(f'{img_src_dir}/{fn}')
            crop = crop_box(img, ann)
            cv2.imwrite(f'{img_tgt_dir}/{crop_fn}',crop)
            labels = extract_labels(ann)
            write_to_classtxt(labels,f'{ann_tgt_dir}',crop_fn,spl)
            create_xml(labels,xml_tgt_dir, crop_fn, crop)


def inv_cls_frq(voc_folder, spl):
    """Compute inverse class frequency to scale loss for multilabel classification."""
    def normalize(frq, n):
        return round(n/frq,5)
    label_files = glob.glob(f'{voc_folder}/*_{spl}.txt')
    label_frq = {}
    for label_file in label_files:
        label = os.path.basename(label_file).split('_')[0]
        with open(label_file) as f:
            label_frq[label] = len(f.readlines())
    
    n = sum(label_frq.values())
    norm_frq = {k: normalize(v,n) for k,v in label_frq.items()}
    GESTURE_CLASSES = ['cooking', 'dancing', 'drinking', 'eating', 'holding the nose', 'painting',
    'peeing', 'playing music', 'praying', 'reading', 'sleeping', 'smoking', 'sniffing', 'textile work',
    'writing', 'none']

    return [norm_frq[cls] for cls in GESTURE_CLASSES]

if __name__ == '__main__':
    main()
