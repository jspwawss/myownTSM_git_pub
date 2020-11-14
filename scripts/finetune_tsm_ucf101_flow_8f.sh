python main.py ucf101 Flow \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.00025 --lr_steps 7 15 --epochs 35 \
     --batch-size 16 -j 16 --dropout 0.7 --consensus_type=conv1d --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --concat=All --extra_temporal_modeling \
     --tune_from=pretrained/TSM_kinetics_Flow_resnet50_shift8_blockres_avg_segment8_e50.pth \
     --suffix=MSTSM_TFDEM_split3