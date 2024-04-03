_base_ = [
    'faster-rcnn_r50_fpn.py',
    '../datasets/sniffyart_person_detection.py',
    '../detection_runtime.py'
]

# val_evaluator = dict(
    # type='CocoMetric',
    # ann_file='/hdd/datasets/dsp/annotations_valid.json',
    # metric='bbox',
    # format_only=False,
    # backend_args=None)
# test_evaluator = dict(
    # type='CocoMetric',
    # ann_file='/hdd/datasets/dsp/annotations_test.json',
    # metric='bbox',
    # format_only=False,
    # backend_args=None)

model=dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
        )
    )
)
