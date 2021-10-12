# _base_ = [
#     './ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py'
# ]

_base_ = [
    '../../_base_/models/ircsn_r152.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict( in_channels = 2, #2
        norm_eval=True, bn_frozen=False, bottleneck_mode='ir', pretrained=None),
    cls_head=dict(
        num_classes=24))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/PartHuman/rawframes_train'
data_root_val = 'data/PartHuman/rawframes_test'
ann_file_train = 'data/PartHuman/annot/parthuman_flow_train_list.txt'
ann_file_val = 'data/PartHuman/annot/parthuman_flow_test_list.txt'
ann_file_test = 'data/PartHuman/annot/parthuman_flow_test_list.txt'

img_norm_cfg = dict(mean=[128, 128], std=[128, 128])

train_pipeline = [
    #dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=8, num_clips=1), #视频一般是300帧，主要调整，clip_len=32, frame_interval=8, num_clips=1
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    #dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=8,
        num_clips=1,
        test_mode=True), #视频一般是300帧，主要调整，clip_len=32, frame_interval=8, num_clips=1
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    #dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4,  #每个GPU视频数量
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        modality='Flow',
        filename_tmpl='flow_{}_{:05d}.jpg',
        start_index=1,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        modality='Flow',
        filename_tmpl='flow_{}_{:05d}.jpg',
        start_index=1,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        modality='Flow',
        filename_tmpl='flow_{}_{:05d}.jpg',
        start_index=1,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

optimizer = dict(
    type='SGD', lr=0.02, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 4 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=40) #warmup_iters
total_epochs = 180

checkpoint_config = dict(interval=5)
work_dir = './work_dirs/ircsn_bnfrozen_r152_32x8x1_180e_parthuman_flow'  # noqa: E501
find_unused_parameters = True
workflow = [('train', 1),('val', 1)]