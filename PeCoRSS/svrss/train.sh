CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --master_port 8120 \
        --nproc_per_node=1 \
        --use_env train.py \
        --cfg config_files/fcn8s.json \
        --dataset CWRdata \
        >train_fcn8s_lowres_pwr_1frame_ce_20240406.log 2>&1