#!/usr/bin/env bash
#SBATCH -p gpu-a100                      # --partition
#SBATCH -t 0-01:00:00                    # --time
#SBATCH -c 4                             # --cpus-per-task
#SBATCH --gres=gpu:a100-sxm4:2

#SBATCH -e /home/yangye/O-LoRA/logs_and_outputs/order_1/logs/train_and_infer1.err
#SBATCH -o /home/yangye/O-LoRA/logs_and_outputs/order_1/logs/train_and_infer1.log


cd "$HOME/O-LoRA" 
# Meta Info
echo $(pwd)
mkdir -p "logs"

echo "Host=$(hostname)  SubmitDir=${SLURM_SUBMIT_DIR}"
echo "JobID=${SLURM_JOB_ID}  ArrayIndex=${SLURM_ARRAY_TASK_ID:-0}"

# Main Script
nvidia-smi topo -m \
| sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' \
| expand -t 8



export PATH="/workspace/envs/MLLMs/$USER/conda_envs/lora/bin:$PATH"

set -exuo pipefail
set -x


export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

port=$(shuf -i25000-30000 -n1)
 
# bash scripts/order_1.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &
deepspeed --num_gpus=2 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path /workspace/data/MLLMs/$USER/initial_model/t5-large \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_1/outputs/1-dbpedia \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order1_round1 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0 \
   --lamda_2 0

sleep 5

 deepspeed --num_gpus=2 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_1/outputs/1-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_1/outputs/2-amazon \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order1_round2 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0 \
   --lamda_2 0

sleep 5

 deepspeed --num_gpus=2 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_1/outputs/2-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_1/outputs/3-yahoo \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order1_round3 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0 \
   --lamda_2 0

sleep 5

 deepspeed --num_gpus=2 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_1/outputs/3-yahoo/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_1/outputs/4-agnews \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order1_round4 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0 \
   --lamda_2 0
echo "Finished".