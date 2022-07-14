_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet21k_bs128.py',
    '../_base_/schedules/imagenet_bs1024_coslr.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(head=dict(num_classes=21843))

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=90)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)
