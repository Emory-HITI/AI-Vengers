#!/bin/bash

cd ../

echo "Running experiment:" "${1}"

slurm_pre="--partition t4v2,t4v1,rtx6000,p100 --gres gpu:1 --mem 40gb -c 4 --exclude gpu080 --job-name ${1} --output /scratch/ssd001/home/haoran/projects/CXR_Bias/logs/${1}_%A.log"

python sweep.py launch \
    --experiment ${1} \
    --output_dir "/scratch/hdd001/home/haoran/cxr_bias/${1}/" \
    --slurm_pre "${slurm_pre}" \
    --command_launcher "slurm" 
