bash ./tools/dist_test.sh ./configs/parthuman_val/csn_arm/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/arm/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_val_ircsn/csn_arm.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman_val/csn_hand/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/hand/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_val_ircsn/csn_hand.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman_val/csn_head/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/head/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_val_ircsn/csn_head.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman_val/csn_hip/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/hip/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_val_ircsn/csn_hip.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman_val/csn_foot/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/foot/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_val_ircsn/csn_foot.pkl
wait

bash ./tools/dist_test.sh ./configs/parthuman_val/csn_leg/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb.py ./work_dirs/leg/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_parthuman_rgb/latest.pth 4 --out ./outputs_val_ircsn/csn_leg.pkl
wait

