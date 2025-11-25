_base_ = [
    'gesture_detection_rn50.py'
]


dataset_type = 'CocoDataset' # cross validation datasets have default coco format (no funky custom sniffyart format)
classes = ('cooking', 'dancing', 'drinking', 'eating', 'holding the nose',
        'peeing', 'playing music', 'praying', 'reading', 'sleeping', 
        'smoking', 'sniffing', 'textile work', 'writing', 'painting', 
        'none')     # still need sensoryart classes


train_dataloader = dict(
    batch_size=2,
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

#max_epochs = 200 # sanity check first
max_epochs = 200
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs)
    # _delete_=True,
    # type='IterBasedTrainLoop',
    # max_iters=100,
    # val_interval=100)