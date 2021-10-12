#!/usr/bin/env bash

cd ../
python build_rawframes_multigpus.py /export/home/data/PartHuman/train_videos/ /export/home/data/PartHuman/train_videos_frames/ --level 2 --flow-type tvl1 --ext mp4 --task both
echo "Raw frames (RGB and tv-l1) Generated"
echo "Raw frames (RGB and tv-l1) Generated"
wait

python fileplusone.py

wait