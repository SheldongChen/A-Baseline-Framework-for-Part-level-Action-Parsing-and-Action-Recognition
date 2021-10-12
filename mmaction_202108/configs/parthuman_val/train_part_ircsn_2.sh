#bash ./tools/dist_train.sh ./configs/parthuman/csn_arm/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py 4
#
#wait
#
#
#bash ./tools/dist_train.sh ./configs/parthuman/csn_hand/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py 4
#
#wait
#
#bash ./tools/dist_train.sh ./configs/parthuman/csn_head/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py 4
#
#wait

bash ./tools/dist_train.sh ./configs/parthuman/csn_hip/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py 4

wait

bash ./tools/dist_train.sh ./configs/parthuman/csn_foot/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py 4

wait

bash ./tools/dist_train.sh ./configs/parthuman/csn_leg/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py 4

wait