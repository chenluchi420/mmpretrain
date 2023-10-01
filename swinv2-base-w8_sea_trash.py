_base_ = [
    'configs/_base_/models/swin_transformer_v2/small_256.py',
    'configs/_base_/datasets/imagenet_bs64_sea_trash.py',
    'configs/_base_/schedules/imagenet_bs1024_adamw_swin.py',
    'configs/_base_/default_runtime.py'
]
