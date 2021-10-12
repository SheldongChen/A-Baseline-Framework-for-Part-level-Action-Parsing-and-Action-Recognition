bash ./tools/dist_train.sh ./configs/parthuman/csn_arm/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py 4

wait

bash ./tools/dist_train.sh ./configs/parthuman/csn_foot/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py 4

wait

bash ./tools/dist_train.sh ./configs/parthuman/csn_hand/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py 4

wait

bash ./tools/dist_train.sh ./configs/parthuman/csn_leg/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py 4

wait
