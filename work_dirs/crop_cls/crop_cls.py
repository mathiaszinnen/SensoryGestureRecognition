FRQS = [
    56.41,
    38.90345,
    10.92159,
    40.29286,
    22.12157,
    90.256,
    451.28,
    19.28547,
    26.54588,
    51.28182,
    132.72941,
    12.00213,
    56.41,
    90.256,
    322.34286,
    1.8185,
]
SNIFFYART_CLASSES = [
    'cooking',
    'dancing',
    'drinking',
    'eating',
    'holding the nose',
    'painting',
    'peeing',
    'playing music',
    'praying',
    'reading',
    'sleeping',
    'smoking',
    'sniffing',
    'textile work',
    'writing',
    'none',
]
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth'
data_preprocessor = dict(
    mean=[
        0,
        0,
        0,
    ],
    num_classes=16,
    std=[
        255,
        255,
        255,
    ],
    to_onehot=True,
    to_rgb=True)
data_root = 'data/crop_cls/VOC2012/'
dataset_type = 'VOC'
default_hooks = dict(
    checkpoint=dict(_scope_='mmpretrain', interval=1, type='CheckpointHook'),
    logger=dict(_scope_='mmpretrain', interval=100, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmpretrain', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmpretrain', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmpretrain', type='IterTimerHook'),
    visualization=dict(
        _scope_='mmpretrain', enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    _scope_='mmpretrain',
    backbone=dict(
        arch='base',
        img_size=384,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth',
            prefix='backbone',
            type='Pretrained'),
        stage_cfgs=dict(block_cfgs=dict(window_size=12)),
        type='SwinTransformer'),
    head=dict(
        in_channels=1024,
        lam=0.1,
        loss=dict(
            loss_weight=0.1,
            pos_weight=[
                56.41,
                38.90345,
                10.92159,
                40.29286,
                22.12157,
                90.256,
                451.28,
                19.28547,
                26.54588,
                51.28182,
                132.72941,
                12.00213,
                56.41,
                90.256,
                322.34286,
                1.8185,
            ],
            type='CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=16,
        num_heads=1,
        type='CSRAClsHead'),
    neck=None,
    type='ImageClassifier')
num_classes = 16
optim_wrapper = dict(
    optimizer=dict(
        _scope_='mmpretrain',
        lr=0.0002,
        momentum=0.9,
        type='SGD',
        weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10))))
param_scheduler = [
    dict(
        _scope_='mmpretrain',
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=1,
        start_factor=1e-07,
        type='LinearLR'),
    dict(
        _scope_='mmpretrain',
        by_epoch=True,
        gamma=0.1,
        step_size=6,
        type='StepLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        _scope_='mmpretrain',
        classes=[
            'cooking',
            'dancing',
            'drinking',
            'eating',
            'holding the nose',
            'painting',
            'peeing',
            'playing music',
            'praying',
            'reading',
            'sleeping',
            'smoking',
            'sniffing',
            'textile work',
            'writing',
            'none',
        ],
        data_root='data/crop_cls/VOC2012/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=448, type='Resize'),
            dict(
                meta_keys=(
                    'sample_idx',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'gt_label_difficult',
                ),
                type='PackInputs'),
        ],
        split='test',
        type='VOC'),
    num_workers=5,
    sampler=dict(_scope_='mmpretrain', shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(_scope_='mmpretrain', type='VOCMultiLabelMetric'),
    dict(_scope_='mmpretrain', average='micro', type='VOCMultiLabelMetric'),
    dict(_scope_='mmpretrain', type='VOCAveragePrecision'),
]
test_pipeline = [
    dict(_scope_='mmpretrain', type='LoadImageFromFile'),
    dict(_scope_='mmpretrain', scale=448, type='Resize'),
    dict(
        _scope_='mmpretrain',
        meta_keys=(
            'sample_idx',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'gt_label_difficult',
        ),
        type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        _scope_='mmpretrain',
        data_root='data/crop_cls/VOC2012/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                crop_ratio_range=(
                    0.7,
                    1.0,
                ),
                scale=448,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='VOC'),
    num_workers=5,
    sampler=dict(_scope_='mmpretrain', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(_scope_='mmpretrain', type='LoadImageFromFile'),
    dict(
        _scope_='mmpretrain',
        crop_ratio_range=(
            0.7,
            1.0,
        ),
        scale=448,
        type='RandomResizedCrop'),
    dict(
        _scope_='mmpretrain',
        direction='horizontal',
        prob=0.5,
        type='RandomFlip'),
    dict(_scope_='mmpretrain', type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        _scope_='mmpretrain',
        classes=[
            'cooking',
            'dancing',
            'drinking',
            'eating',
            'holding the nose',
            'painting',
            'peeing',
            'playing music',
            'praying',
            'reading',
            'sleeping',
            'smoking',
            'sniffing',
            'textile work',
            'writing',
            'none',
        ],
        data_root='data/crop_cls/VOC2012/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=448, type='Resize'),
            dict(
                meta_keys=(
                    'sample_idx',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'gt_label_difficult',
                ),
                type='PackInputs'),
        ],
        split='val',
        type='VOC'),
    num_workers=5,
    sampler=dict(_scope_='mmpretrain', shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(_scope_='mmpretrain', type='VOCMultiLabelMetric'),
    dict(_scope_='mmpretrain', average='micro', type='VOCMultiLabelMetric'),
    dict(_scope_='mmpretrain', type='VOCAveragePrecision'),
]
vis_backends = [
    dict(_scope_='mmpretrain', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmpretrain',
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/crop_cls'
