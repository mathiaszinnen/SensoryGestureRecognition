_base_ = ['mmpretrain::csra/resnet101-csra_1xb16_voc07-448px.py']

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth'

SNIFFYART_CLASSES = [
    'cooking', 'dancing', 'drinking', 'eating', 'holding the nose', 
    'painting', 'peeing','playing music', 'praying', 'reading', 
    'sleeping', 'smoking', 'sniffing', 'textile work','writing', 
    'none'
]
#FRQS = [0.01773, 0.0257, 0.09156, 0.02482, 0.0452, 0.01108, 0.00222, 0.05185, 
        #0.03767, 0.0195, 0.00753, 0.08332, 0.01773, 0.01108, 0.0031, 0.5499]
FRQS = [56.41, 38.90345, 10.92159, 40.29286, 22.12157, 90.256, 451.28, 19.28547, 26.54588, 51.28182, 132.72941, 12.00213, 56.41, 90.256, 322.34286, 1.8185]
num_classes = len(SNIFFYART_CLASSES)

default_hooks = dict(checkpoint=dict(
    interval=50,
    save_best='multi-label/f1-score',
    rule='greater'
))

train_cfg = dict(
    max_epochs = 50, val_interval=1
)

data_preprocessor = dict(
    num_classes = num_classes
)

data_root = 'data/crop_cls/VOC2012/'

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
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        arch='base',
        img_size=384,
        stage_cfgs=dict(block_cfgs=dict(window_size=12)),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint=checkpoint
        )
    ),
    head = dict(
        num_classes=num_classes,
        in_channels=1024,
        loss=dict(pos_weight=FRQS, loss_weight=0.1)
    )
)

test_evaluator = [
    dict(_scope_='mmpretrain', type='VOCMultiLabelMetric'),
    dict(_scope_='mmpretrain', type='VOCMultiLabelMetric', average=None),
    dict(_scope_='mmpretrain', average='micro', type='VOCMultiLabelMetric'),
    dict(_scope_='mmpretrain', type='VOCAveragePrecision'),
    # dict(_scope_='mmpretrain', type='ConfusionMatqrix'),
]