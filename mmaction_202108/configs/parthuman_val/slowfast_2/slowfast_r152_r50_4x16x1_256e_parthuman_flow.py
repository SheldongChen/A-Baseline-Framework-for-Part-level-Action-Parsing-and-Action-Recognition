model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,  # tau
        speed_ratio=8,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=152,
            in_channels=2,  # 2
            with_pool2=False,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            in_channels=2,  # 2
            with_pool2=False,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=24,
        spatial_type='avg',
        dropout_ratio=0.5))
train_cfg = None
test_cfg = dict(average_clips='prob', max_testing_views=8)

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
    dict(type='SampleFrames', clip_len=32, frame_interval=8, num_clips=1),
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
        test_mode=True),
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
    videos_per_gpu=8,
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
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        modality='Flow',
        filename_tmpl='flow_{}_{:05d}.jpg',
        start_index=1,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.006, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 4 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=34)
total_epochs = 256
checkpoint_config = dict(interval=4)
workflow = [('train', 1),('val',1)]
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        #    dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
checkpoint_config = dict(interval=5)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/slowfast_r152_r50_4x16x1_256e_parthuman_flow'
load_from = None
resume_from = None
find_unused_parameters = False
