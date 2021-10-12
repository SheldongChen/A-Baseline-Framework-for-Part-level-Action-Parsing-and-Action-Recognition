bash ./tools/dist_train.sh ./configs/yolof_part_only/yolof_r101_c5_8x8_1x_arm.py  4
wait

bash ./tools/dist_train.sh ./configs/yolof_part_only/yolof_r101_c5_8x8_1x_foot.py  4
wait

bash ./tools/dist_train.sh ./configs/yolof_part_only/yolof_r101_c5_8x8_1x_hand.py  4
wait

bash ./tools/dist_train.sh ./configs/yolof_part_only/yolof_r101_c5_8x8_1x_head.py  4
wait

bash ./tools/dist_train.sh ./configs/yolof_part_only/yolof_r101_c5_8x8_1x_hip.py  4
wait

bash ./tools/dist_train.sh ./configs/yolof_part_only/yolof_r101_c5_8x8_1x_leg.py  4
wait