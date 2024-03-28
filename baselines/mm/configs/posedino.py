_base_ = [
    'datasets/sniffyart_gesture_detection.py', 
    'detection_runtime.py'
]

custom_imports = dict(
    imports=[
        'gesture_recognition.models.posedino',
        'gesture_recognition.datasets.sniffyart_multitask',
        'gesture_recognition.datasets.transforms'
    ],
    allow_failed_imports=False
)

#checkpoint = '/home/woody/iwi5/iwi5093h/models/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth'
checkpoint = '/localhome/mathias/hdd/sniffyart/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth'

num_levels = 4

model = dict(
    type='POSEDINO',
    pose_weight=1000.,
    det_weight=1.,
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='MultiTaskDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
        pose_preprocessor = dict(
            _scope_='mmpose',
            type='PoseDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True)
        ),
    backbone=dict(
        # swin base has embed dims: 128, depths: [2,2,18,2], and num_heads: [4,8,16,32], see https://github.com/open-mmlab/mmpretrain/blob/17a886cb5825cd8c26df4e65f7112d404b99fe12/mmpretrain/models/backbones/swin_transformer.py#L275
        type='SwinTransformer',
        embed_dims=128,
        depths=[2,2,18,2],
        num_heads=[4,8,16,32],
        pretrain_img_size=384,
        window_size=12,
        init_cfg=dict(type='Pretrained', 
                      checkpoint=checkpoint,
                      prefix='backbone')
        ),
    neck=dict(
        type='ChannelMapper',
        #in_channels=[512, 1024, 2048],
        in_channels=[128,256,512,1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_levels),
    pose_neck=dict(
        type='mmpose.FeatureMapProcessor',
        concat=True
    ),
    pose_heads=dict(
        _scope_='mmpose',
        type='DEKRHead',
        in_channels=1920,
        num_keypoints=17,
        heatmap_loss=dict(type='KeypointMSELoss', use_target_weight=True),
        displacement_loss=dict(
            type='SoftWeightSmoothL1Loss',
            use_target_weight=True,
            supervise_empty=False,
            beta=1 / 9,
            loss_weight=0.002,
        ),        
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=num_levels,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=num_levels,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHead',
        num_classes=16,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR


pose_codec = dict(
    type='SPR',
    input_size=(512, 512),
    heatmap_size=(128, 128),
    sigma=(4, 2),
    minimal_diagonal_length=32**0.5,
    generate_keypoint_heatmaps=True,
    decode_max_instances=30)
pose_target_generator=dict(
    type='mmpose.GenerateTarget',
    encoder=pose_codec
)



# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadMultiTaskAnnotations', 
         with_kpts=True),
    dict(type='MultiTaskResize',
         scale=(512,512),
         keep_ratio=False),
    dict(type='mmpose.GenerateTarget', encoder=pose_codec),
    dict(type='FixedBottomupGetHeatmapMask'),
    dict(type='PackMultiTaskInputs',
         posepacker=dict(
             type='mmpose.PackPoseInputs',
         ),
         detpacker=dict(
             type='PackDetInputs'
         ))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadMultiTaskAnnotations', 
         with_kpts=True),
    dict(type='MultiTaskResize',
         scale=(512,512),
         keep_ratio=False),
    dict(type='PackMultiTaskInputs',
         posepacker=dict(
             type='mmpose.PackPoseInputs',
         ),
         detpacker=dict(
             type='PackDetInputs'
         ))
]
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='SniffyArtMultiTask',
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        type='SniffyArtMultiTask',
        pipeline=test_pipeline
    )
)
test_dataloader=dict(
    dataset=dict(
        pipeline=test_pipeline
    )
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 200
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[150, 180],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)
