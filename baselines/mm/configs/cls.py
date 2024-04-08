_base_ = ['crop_cls.py']

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth'

SNIFFYART_CLASSES = [
    'cooking', 'dancing', 'drinking', 'eating', 'holding the nose', 
    'painting', 'peeing','playing music', 'praying', 'reading', 
    'sleeping', 'smoking', 'sniffing', 'textile work','writing'
]
FRQS =[25.06667, 23.5, 4.7, 20.32432, 10.74286, 34.18182, 150.4, 7.91579, 15.04, 19.78947, 62.66667, 5.22222, 21.48571, 50.13333, 107.42857] 
num_classes = len(SNIFFYART_CLASSES)

data_preprocessor = dict(
    num_classes = num_classes
)

data_root = 'data/cls/VOC2012/'

train_dataloader = dict(
    dataset = dict(
        data_root=data_root,
        split='train',
        classes=SNIFFYART_CLASSES
    )
)

val_dataloader = dict(
    dataset = dict(
        data_root=data_root,
        split='val',
        classes=SNIFFYART_CLASSES
    )
)
test_dataloader = dict(
    dataset = dict(
        data_root=data_root,
        split='test',
        classes=SNIFFYART_CLASSES
    )
)


model = dict(
    head = dict(
        num_classes=num_classes,
        loss=dict(pos_weight=FRQS, loss_weight=0.1)
    )
)