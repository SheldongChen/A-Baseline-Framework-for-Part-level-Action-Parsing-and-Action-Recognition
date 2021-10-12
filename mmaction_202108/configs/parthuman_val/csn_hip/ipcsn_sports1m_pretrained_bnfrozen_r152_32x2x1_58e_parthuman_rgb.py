_base_ = [
    './ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py'
]

# model settings
model = dict(
    backbone=dict(
        norm_eval=True,
        bn_frozen=True,
        bottleneck_mode='ip',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/csn/ipcsn_from_scratch_r152_sports1m_20210617-7a7cc5b9.pth'  # noqa: E501
    ),
    cls_head=dict(
        num_classes=78,
    )
)
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy'])
workflow = [('train', 1),('val', 1)]
work_dir = './work_dirs/hip/ipcsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb'  # noqa: E501
