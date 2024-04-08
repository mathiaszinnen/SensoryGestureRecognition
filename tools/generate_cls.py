import json
import glob
import os, shutil
import xml.etree.cElementTree as ET
import cv2
from tqdm import tqdm


def extract_img_labels(img_anns):
    gestures = set() 
    for ann in img_anns:
        gestures.update(ann['gestures'])
    return list(gestures)

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

        for img in tqdm(coco['images']):
            fn = img["file_name"]
            img_anns = [ann for ann in coco['annotations'] if ann['image_id'] == img['id']]
            img_labels = extract_img_labels(img_anns)
            # if len(img_labels) == 0:
                # img_labels = ['none']
            shutil.copyfile(f'{img_src_dir}/{fn}',
                            f'{img_tgt_dir}/{fn}')
            write_to_classtxt(img_labels, ann_tgt_dir, fn, spl)
            cv2img = cv2.imread(f'{img_src_dir}/{fn}')
            create_xml(img_labels, xml_tgt_dir, fn, cv2img)


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
    'writing']

    return [norm_frq[cls] for cls in GESTURE_CLASSES]


if __name__ == '__main__':
    # main()
    clsfrqs = inv_cls_frq('data/cls/VOC2012/ImageSets/Main', 'train')
    print(clsfrqs)