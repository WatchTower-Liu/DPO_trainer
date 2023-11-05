# CUDA_VISIBLE_DEVICES=5,6 accelerate launch --multi_gpu DPO_trl.py \
CUDA_VISIBLE_DEVICES=5 python DPO_trl.py \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--num_train_epochs 1 \
--save_steps 500 \
--save_total_limit 5 \
--learning_rate 5e-7 \
--seed 42 \
--ddp_find_unused_parameters=False \
--remove_unused_columns false \
--logging_steps 10 \
--output_dir ./weights/DPO_BC
