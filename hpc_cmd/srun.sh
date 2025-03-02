#!/bin/bash

#srun --jobid=698933 --pty bash -i

# 檢查是否提供了 Job ID
if [ -z "$1" ]; then
    echo "用法: $0 <jobid>"
    exit 1
fi

# 取得 Job ID
JOB_ID=$1

# 執行 srun 登入指定的 Job ID
srun --jobid=$JOB_ID nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
 
#srun --jobid=$JOB_ID --pty bash -i
