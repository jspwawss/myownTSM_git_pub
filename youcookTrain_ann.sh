#MSTSM+TFDEM
#CUDA_LAUNCH_BLOCKING=1  
python3 main_ann.py youcook RGB \
	--arch resnet50 --num_segments 8 \
	--store_name=ann\
	--gd 20 --lr 0.00025 --lr_steps 7 15 --epochs 20 \
	--batch-size 16 -j 16 --dropout 0.7 --consensus_type=conv1d --eval-freq=1 \
	--shift --shift_div=8 --shift_place=blockres --concat=All  \
	--tune_from=/home/ubuntu/backup_kevin/myownTSM_git/pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth \
	--suffix=v2 --loss_type=CrossEntropyLoss --clipnums=500
	2>err.log

#MSTSM+TFDEM+prune
# python3 main.py ucf101 RGB \
# 	--arch resnet50 --num_segments 8 \
# 	--gd 20 --lr 0.0003 --lr_steps 5 20 --epochs 35 \
# 	--batch-size 16 -j 16 --dropout 0.7 --consensus_type=conv1d --eval-freq=1 \
# 	--shift --shift_div=8 --shift_place=blockres --concat=All --extra_temporal_modeling --prune=inout\
# 	--tune_from=/home/u9210700/Saved_Models/TSM_ucf101_RGB_resnet50_shift8_blockres_concatAll_conv1d_lr0.00025_dropout0.70_wd0.00050_batch16_segment8_e35_extraLSTM_finetune_MSTSM_TFDEM_split3/ckpt.best.pth.tar \
# 	--suffix=MSTSM_TFDEM_prune_split3
