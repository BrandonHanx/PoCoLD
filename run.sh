export MODEL_NAME="/storage/local/hanxiao/models/stable-diffusion-v1-5"
export TRAIN_DIR="datasets/DeepFashionPose"
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
	  --mixed_precision="fp16" --main_process_port=29500 --num_processes=4 train.py \
	  --pretrained_model_name_or_path=$MODEL_NAME \
	  --train_data_dir=$TRAIN_DIR --load_from_disk \
	  --height=512 --width=384 \
      --do_cfg --from_scratch --use_constraint_penalty \
	  --train_batch_size=8 \
	  --gradient_accumulation_steps=1 \
	  --max_train_steps=600000 \
	  --learning_rate=5e-05 \
	  --max_grad_norm=1 \
	  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=10 --lr_x=1 \
	  --output_dir="models/final" \
      --show_steps=1000 \
      --save_steps=10000
