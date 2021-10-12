bash ./tools/dist_test.sh ./configs/parthuman/csn_arm/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/arm/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_part/csn_arm.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman/csn_hand/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/hand/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_part/csn_hand.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman/csn_head/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/head/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_part/csn_head.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman/csn_hip/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/hip/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_part/csn_hip.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman/csn_foot/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/foot/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_part/csn_foot.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman/csn_leg/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/leg/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_part/csn_leg.pkl
wait

