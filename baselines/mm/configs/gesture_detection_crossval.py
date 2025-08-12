_base_ = [
    'gesture_detection.py'
]


dataset_type = 'CocoDataset' # cross validation datasets have default coco format (no funky custom sniffyart format)
classes = ('cooking', 'dancing', 'drinking', 'eating', 'holding the nose',
        'peeing', 'playing music', 'praying', 'reading', 'sleeping', 
        'smoking', 'sniffing', 'textile work', 'writing', 'painting', 
        'none')     # still need sensoryart classes

train_dataloader = dict(
    dataset = dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017')
    )
)
val_dataloader = dict(
    dataset = dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017')
    )
)
test_dataloader = dict(
    dataset = dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        ann_file='annotations/person_keypoints_test2017.json',
        data_prefix=dict(img='test2017')
    )
)

val_evaluator = dict(
    ann_file = 'data/annotations/person_keypoints_val2017.json'
)
test_evaluator = dict(
    ann_file = 'data/annotations/person_keypoints_test2017.json'
)

max_epochs = 1 # sanity check first
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs)