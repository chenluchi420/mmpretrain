# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.75, direction='horizontal'),
    dict(type='RandomFlip', prob=0.75, direction='vertical'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='ColorJitter',brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, backend='pillow'),
    dict(type='PackInputs'),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/home/extra_space_1/sea_trash/res_used_data/train',
        # split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/home/extra_space_1/sea_trash/res_used_data/val',
        # split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
