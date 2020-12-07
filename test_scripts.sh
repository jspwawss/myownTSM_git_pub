#RGB_checkpoint_path="/home/share/UCF101/Saved_Models/TSM_ucf101_RGB_resnet50_shift8_blockres_concatAll_conv1d_lr0.00030_dropout0.70_wd0.00050_batch16_segment8_e35_extraLSTM_finetune_MSTSM_TFDEM_prune_split2/"
RGB_checkpoint_path="/home/ubuntu/backup_kevin/myownTSM/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_concatAll_conv1d_lr0.00025_dropout0.70_wd0.00050_batch16_segment8_e35_extra_finetune_MSTSM_TFDEM_split3/"
#RGB_checkpoint_path="/home/ubuntu/backup_kevin/myownTSM/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_concatAll_conv1d_lr0.00025_dropout0.70_wd0.00050_batch16_segment8_e35_extra_finetune_MSTSM_TFDEM_prune_split3/"
python3 test_models.py ucf101 \
    --weights=${RGB_checkpoint_path}/ckpt.best.pth.tar \
    --test_list=/home/share/UCF101/file_list/ucf101_rgb_val_split_3.txt \
    --test_segments=8 --test_crops=1 --full_res --wholeFrame --softmax \

    --batch_size=1 \
    --save-scores=${RGB_checkpoint_path}/ucf101_scores 2> test_err.log
    
#Flow_checkpoint_path=""
#python test_models.py ucf101 \
#    --weights=${Flow_checkpoint_path}/ckpt.best.pth.tar \
#    --test_list=/home/share/UCF101/file_list/ucf101_flow_val_split_3.txt \
#    --test_segments=8 --test_crops=1 --full_res --twice_sample --softmax \
#    --batch_size=16 \
#    --save-scores=${Flow_checkpoint_path}/ucf101_scores 2> test_err.log


#python combine.py --RGB ${RGB_checkpoint_path}/ucf101_scores.npz --wRGB 1 \
#--Depth ${Flow_checkpoint_path}/ucf101_scores.npz --wDepth 1

#python combine.py --RGB ${RGB_checkpoint_path}/ucf101_scores.npz --wRGB 1 \
#--Depth ${Flow_checkpoint_path}/ucf101_scores.npz --wDepth 2

#python combine.py --RGB ${RGB_checkpoint_path}/ucf101_scores.npz --wRGB 2 \
#--Depth ${Flow_checkpoint_path}/ucf101_scores.npz --wDepth 1







