#!/bin/bash

# 取得所有 job_id，並對每個 job_id 進行處理
squeue -u c00cjz00 --format="%.18i %.6P %.24j %.8u %.2t %.10M %.6D %R" | tail -n +2 | while read line; do
    # 提取每行的 job_id（假設 job_id 是第一個欄位）
    JOB_ID=$(echo $line | awk '{print $1}')
    NAME_ID=$(echo $line | awk '{print $3}')
	    
    # 使用 job_id 執行 srun 命令
    echo "echo \"## $NAME_ID\""
    echo srun --jobid=$JOB_ID nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
    echo "echo \"    \""
    echo sleep 1
done

