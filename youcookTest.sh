RGB_checkpoint_path="/home/ubuntu/backup_kevin/myownTSM_git/checkpoint/TSM_youcook_RGB_resnet50_shift8_blockres_concatAll_conv1d_lr0.00025_dropout0.70_wd0.00050_batch16_segment8_e50_finetune_clip500_slice_v3/"
#RGB_checkpoint_path="/home/ubuntu/backup_kevin/myownTSM/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_concatAll_conv1d_lr0.00025_dropout0.70_wd0.00050_batch16_segment8_e35_extra_finetune_MSTSM_TFDEM_prune_split3/"
python3 test_models_v2.py youcook \
    --weights=${RGB_checkpoint_path}/ckpt_{}.pth.tar \
    --test_list=/home/share/UCF101/file_list/ucf101_rgb_val_split_3.txt \
    --test_segments=8 --test_crops=1 --full_res --wholeFrame --softmax \
    --shift --shift_div=8 --shift_place=blockres --concat=All  \
    --batch_size=1 --clipnums=500 --epoch=19\
    --save-scores=${RGB_checkpoint_path}
    2> test_err.log
    