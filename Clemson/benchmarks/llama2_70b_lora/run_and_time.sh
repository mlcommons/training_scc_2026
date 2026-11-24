#!/bin/bash
source $HOME/scc/env.sh
NAME=llama-$SLURM_JOB_ID-$SLURM_NODEID
SEED=${SEED:?SEED must be set}; MAXSTEPS=${MAXSTEPS:-1024}; EVALSTEPS=${EVALSTEPS:-48}
trap 'docker rm -f $NAME >/dev/null 2>&1' EXIT TERM INT
echo "$(hostname -s): node_rank=$SLURM_NODEID master=$MASTER seed=$SEED eval_bs=$EVAL_BS script=$TRAIN_SCRIPT"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed "s/^/$(hostname -s) other GPU process: /"
docker run --rm --name $NAME --device nvidia.com/gpu=all --network=host --ipc=host \
  --device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1 \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  -e NCCL_NET_GDR_LEVEL=SYS -e NCCL_SOCKET_IFNAME=$IFACE -e NCCL_IB_HCA=$HCAS \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $REPO:/workspace/ft-llm \
  -v $DATASET_DIR:/workspace/ft-llm/dataset:ro \
  -v $MODEL_DIR:/workspace/ft-llm/models/llama-v2-fused-qkv:ro \
  -v $RESULTS_DIR:/workspace/ft-llm/results \
  -w /workspace/ft-llm $IMAGE \
  accelerate launch --config_file configs/olympus_2node_final.yaml \
    --machine_rank $SLURM_NODEID --main_process_ip $MASTER --main_process_port 29500 \
    scripts/$TRAIN_SCRIPT \
    --dataset_path ./dataset --model_path ./models/llama-v2-fused-qkv \
    --max_seq_length 8192 --bf16 True --logging_steps 24 --eval_steps $EVALSTEPS \
    --per_device_eval_batch_size $EVAL_BS \
    --output_dir ./results/llama-70b_r16_${SEED}_$SLURM_JOB_ID \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine --learning_rate 4e-4 --weight_decay 0.0001 \
    --warmup_ratio 0 --max_grad_norm 0.3 --use_gradient_checkpointing True \
    --target_eval_loss 0.925 --use_peft_lora True --lora_r 16 --lora_alpha 32 \
    --lora_dropout 0.1 --max_steps $MAXSTEPS --use_flash_attn --seed $SEED \
    --lora_target_modules 'qkv_proj,o_proj' &
wait $!
