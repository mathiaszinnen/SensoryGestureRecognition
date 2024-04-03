_base_ = ['faster-rcnn_r50_fpn_1x_dsp.py']

model=dict(
  backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
        )
    )
)
