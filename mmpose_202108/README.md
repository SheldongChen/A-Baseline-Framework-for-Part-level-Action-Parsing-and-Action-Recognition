<div align="center">
    <img src="resources/mmpose-logo.png" width="400"/>
</div>

## Introduction

English | [简体中文](README_CN.md)

[![Documentation](https://readthedocs.org/projects/mmpose/badge/?version=latest)](https://mmpose.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmpose/workflows/build/badge.svg)](https://github.com/open-mmlab/mmpose/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmpose/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmpose)
[![PyPI](https://img.shields.io/pypi/v/mmpose)](https://pypi.org/project/mmpose/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)

MMPose is an open-source toolbox for pose estimation based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

The master branch works with **PyTorch 1.3+**.

https://user-images.githubusercontent.com/15977946/124654387-0fd3c500-ded1-11eb-84f6-24eeddbf4d91.mp4

### Major Features

- **Support diverse tasks**

  We support a wide spectrum of mainstream pose analysis tasks in current research community, including 2d multi-person human pose estimation, 2d hand pose estimation, 2d face landmark detection, 133 keypoint whole-body human pose estimation, 3d human mesh recovery, fashion landmark detection and animal pose estimation.
  See [demo.md](demo/README.md) for more information.

- **Higher efficiency and higher accuracy**

  MMPose implements multiple state-of-the-art (SOTA) deep learning models, including both top-down & bottom-up approaches. We achieve faster training speed and higher accuracy than other popular codebases, such as [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).
  See [benchmark.md](docs/benchmark.md) for more information.

- **Support for various datasets**

  The toolbox directly supports multiple popular and representative datasets, COCO, AIC, MPII, MPII-TRB, OCHuman etc.
  See [data_preparation.md](docs/data_preparation.md) for more information.

- **Well designed, tested and documented**

  We decompose MMPose into different components and one can easily construct a customized
  pose estimation framework by combining different modules.
  We provide detailed documentation and API reference, as well as unittests.

## [Model Zoo](https://mmpose.readthedocs.io/en/latest/modelzoo.html)

Supported algorithms:

<details open>
<summary>(click to collapse)</summary>

- [x] [DeepPose](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#deeppose-cvpr-2014) (CVPR'2014)
- [x] [CPM](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#cpm-cvpr-2016) (CVPR'2016)
- [x] [Hourglass](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#hourglass-eccv-2016) (ECCV'2016)
- [x] [MSPN](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#mspn-arxiv-2019) (ArXiv'2019)
- [x] [RSN](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#rsn-eccv-2020) (ECCV'2020)
- [x] [SimpleBaseline2D](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#simplebaseline2d-eccv-2018) (ECCV'2018)
- [x] [HRNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#hrnet-cvpr-2019) (CVPR'2019)
- [x] [HRNetv2](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#hrnetv2-tpami-2019) (TPAMI'2019)
- [x] [LiteHRNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#litehrnet-cvpr-2021) (CVPR'2021)
- [x] [SCNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#scnet-cvpr-2020) (CVPR'2020)
- [x] [Associative Embedding](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#associative-embedding-nips-2017) (NeurIPS'2017)
- [x] [HigherHRNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#higherhrnet-cvpr-2020) (CVPR'2020)
- [x] [HMR](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#hmr-cvpr-2018) (CVPR'2018)
- [x] [SimpleBaseline3D](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#simplebaseline3d-iccv-2017) (ICCV'2017)
- [x] [InterNet](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#internet-eccv-2020) (ECCV'2020)
- [x] [VideoPose3D](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#videopose3d-cvpr-2019) (CVPR'2019)
- [x] [ViPNAS](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#vipnas-cvpr-2021) (CVPR'2021)

</details>

Supported techniques:

<details open>
<summary>(click to collapse)</summary>

- [x] [Wingloss](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#wingloss-cvpr-2018) (CVPR'2018)
- [x] [DarkPose](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#darkpose-cvpr-2020) (CVPR'2020)
- [x] [UDP](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#udp-cvpr-2020) (CVPR'2020)
- [x] [FP16](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#fp16-arxiv-2017) (ArXiv'2017)
- [x] [Albumentations](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#albumentations-information-2020) (Information'2020)

</details>

Supported [datasets](https://mmpose.readthedocs.io/en/latest/datasets.html):

<details open>
<summary>(click to collapse)</summary>

- [x] [COCO](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#coco-eccv-2014) \[[homepage](http://cocodataset.org/)\] (ECCV'2014)
- [x] [COCO-WholeBody](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#coco-wholebody-eccv-2020) \[[homepage](https://github.com/jin-s13/COCO-WholeBody/)\] (ECCV'2020)
- [x] [MPII](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#mpii-cvpr-2014) \[[homepage](http://human-pose.mpi-inf.mpg.de/)\] (CVPR'2014)
- [x] [MPII-TRB](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#mpii-trb-iccv-2019) \[[homepage](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)\] (ICCV'2019)
- [x] [AI Challenger](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#ai-challenger-arxiv-2017) \[[homepage](https://github.com/AIChallenger/AI_Challenger_2017)\] (ArXiv'2017)
- [x] [OCHuman](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#ochuman-cvpr-2019) \[[homepage](https://github.com/liruilong940607/OCHumanApi)\] (CVPR'2019)
- [x] [CrowdPose](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#crowdpose-cvpr-2019) \[[homepage](https://github.com/Jeff-sjtu/CrowdPose)\] (CVPR'2019)
- [x] [PoseTrack18](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#posetrack18-cvpr-2018) \[[homepage](https://posetrack.net/users/download.php)\] (CVPR'2018)
- [x] [MHP](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#mhp-acm-mm-2018) \[[homepage](https://lv-mhp.github.io/dataset)\] (ACM MM'2018)
- [x] [sub-JHMDB](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#jhmdb-iccv-2013) \[[homepage](http://jhmdb.is.tue.mpg.de/dataset)\] (ICCV'2013)
- [x] [Human3.6M](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#human3-6m-tpami-2014) \[[homepage](http://vision.imar.ro/human3.6m/description.php)\] (TPAMI'2014)
- [x] [300W](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#300w-imavis-2016) \[[homepage](https://ibug.doc.ic.ac.uk/resources/300-W/)\] (IMAVIS'2016)
- [x] [WFLW](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#wflw-cvpr-2018) \[[homepage](https://wywu.github.io/projects/LAB/WFLW.html)\] (CVPR'2018)
- [x] [AFLW](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#aflw-iccvw-2011) \[[homepage](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)\] (ICCVW'2011)
- [x] [COFW](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#cofw-iccv-2013) \[[homepage](http://www.vision.caltech.edu/xpburgos/ICCV13/)\] (ICCV'2013)
- [x] [OneHand10K](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#onehand10k-tcsvt-2019) \[[homepage](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)\] (TCSVT'2019)
- [x] [FreiHand](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#freihand-iccv-2019) \[[homepage](https://lmb.informatik.uni-freiburg.de/projects/freihand/)\] (ICCV'2019)
- [x] [RHD](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#rhd-iccv-2017) \[[homepage](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)\] (ICCV'2017)
- [x] [CMU Panoptic HandDB](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#cmu-panoptic-handdb-cvpr-2017) \[[homepage](http://domedb.perception.cs.cmu.edu/handdb.html)\] (CVPR'2017)
- [x] [InterHand2.6M](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#interhand2-6m-eccv-2020) \[[homepage](https://mks0601.github.io/InterHand2.6M/)\] (ECCV'2020)
- [x] [DeepFashion](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#deepfashion-cvpr-2016) \[[homepage](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)\] (CVPR'2016)
- [x] [Animal-Pose](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#animal-pose-iccv-2019) \[[homepage](https://sites.google.com/view/animal-pose/)\] (ICCV'2019)
- [x] [Horse-10](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#horse-10-wacv-2021) \[[homepage](http://www.mackenziemathislab.org/horse10)\] (WACV'2021)
- [x] [MacaquePose](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#macaquepose-biorxiv-2020) \[[homepage](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html)\] (bioRxiv'2020)
- [x] [Vinegar Fly](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#vinegar-fly-nature-methods-2019) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Nature Methods'2019)
- [x] [Desert Locust](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#desert-locust-elife-2019) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [Grévy’s Zebra](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#grevys-zebra-elife-2019) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [ATRW](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#atrw-acm-mm-2020) \[[homepage](https://cvwc2019.github.io/challenge.html)\] (ACM MM'2020)

</details>

Supported backbones:

<details>
<summary>(click to expand)</summary>

- [x] [AlexNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#alexnet-neurips-2012) (NeurIPS'2012)
- [x] [VGG](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#vgg-iclr-2015) (ICLR'2015)
- [x] [ResNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#resnet-cvpr-2016) (CVPR'2016)
- [x] [ResNetV1D](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#resnetv1d-cvpr-2019) (CVPR'2019)
- [x] [ResNeSt](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#resnest-arxiv-2020) (ArXiv'2020)
- [x] [ResNext](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#resnext-cvpr-2017) (CVPR'2017)
- [x] [SEResNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#seresnet-cvpr-2018) (CVPR'2018)
- [x] [ShufflenetV1](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#shufflenetv1-cvpr-2018) (CVPR'2018)
- [x] [ShufflenetV2](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#shufflenetv2-eccv-2018) (ECCV'2018)
- [x] [MobilenetV2](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#mobilenetv2-cvpr-2018) (CVPR'2018)

</details>

Results and models are available in the *README.md* of each method's config directory.
A summary can be found in the [**model zoo**](https://mmpose.readthedocs.io/en/latest/modelzoo.html) page.
We will keep up with the latest progress of the community, and support more popular algorithms and frameworks.

If you have any feature requests, please feel free to leave a comment in [Issues](https://github.com/open-mmlab/mmpose/issues/9).

## Benchmark

We demonstrate the superiority of our MMPose framework in terms of speed and accuracy on the standard COCO keypoint detection benchmark.

| Model | Input size| MMPose (s/iter) | [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) (s/iter) | MMPose (mAP) | [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) (mAP) |
| :--- | :---------------: | :---------------: |:--------------------: | :----------------------------: | :-----------------: |
| resnet_50  | 256x192  | **0.28** | 0.64 | **0.718** | 0.704 |
| resnet_50  | 384x288  | **0.81** | 1.24 | **0.731** | 0.722 |
| resnet_101 | 256x192  | **0.36** | 0.84 | **0.726** | 0.714 |
| resnet_101 | 384x288  | **0.79** | 1.53 | **0.748** | 0.736 |
| resnet_152 | 256x192  | **0.49** | 1.00 | **0.735** | 0.720 |
| resnet_152 | 384x288  | **0.96** | 1.65 | **0.750** | 0.743 |
| hrnet_w32  | 256x192  | **0.54** | 1.31 | **0.746** | 0.744 |
| hrnet_w32  | 384x288  | **0.76** | 2.00 | **0.760** | 0.758 |
| hrnet_w48  | 256x192  | **0.66** | 1.55 | **0.756** | 0.751 |
| hrnet_w48  | 384x288  | **1.23** | 2.20 | **0.767** | 0.763 |

More details about the benchmark are available on [benchmark.md](docs/benchmark.md).

## Installation

Please refer to [install.md](docs/install.md) for installation.

## Data Preparation

Please refer to [data_preparation.md](docs/data_preparation.md) for a general knowledge of data preparation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMPose.
There are also tutorials:

- [learn about configs](docs/tutorials/0_config.md)
- [finetune model](docs/tutorials/1_finetune.md)
- [add new dataset](docs/tutorials/2_new_dataset.md)
- [customize data pipelines](docs/tutorials/3_data_pipeline.md)
- [add new modules](docs/tutorials/4_new_modules.md)
- [export a model to ONNX](docs/tutorials/5_export_model.md)
- [customize runtime settings](docs/tutorials/6_customize_runtime.md)

## FAQ

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```

## Contributing

We appreciate all contributions to improve MMPose. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMPose is an open source project that is contributed by researchers and engineers from various colleges and companies.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new models.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab next-generation toolbox for generative models.
