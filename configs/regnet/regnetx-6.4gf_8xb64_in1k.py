_base_ = ['./regnetx-400mf_8xb128_in1k.py']

# model settings
model = dict(
    backbone=dict(type='RegNet', arch='regnetx_6.4gf'),
    head=dict(in_channels=1624, ))

# dataset settings
train_dataloader = dict(batch_size=64)

# schedule settings
# for batch_size 512, use lr = 0.4
optim_wrapper = dict(optimizer=dict(lr=0.4))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=512)
