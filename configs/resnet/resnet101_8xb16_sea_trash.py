_base_ = [
    '../_base_/models/resnest101.py',
    '../_base_/datasets/imagenet_bs64_sea_trash.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]
