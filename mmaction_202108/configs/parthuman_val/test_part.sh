bash ./tools/dist_test.sh ./configs/parthuman_val/csn_arm/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/arm/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_val_ipcsn/csn_arm.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman_val/csn_hand/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/hand/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_val_ipcsn/csn_hand.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman_val/csn_head/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/head/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_val_ipcsn/csn_head.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman_val/csn_hip/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/hip/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_val_ipcsn/csn_hip.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman_val/csn_foot/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/foot/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_val_ipcsn/csn_foot.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman_val/csn_leg/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/leg/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_val_ipcsn/csn_leg.pkl
wait

