#!/bin/bash
#SBATCH --job-name=llama-3.2-3B-base # 設定作業名稱為 "llama-3.2-3B-base"
#SBATCH --partition=gp4d             # 指定使用 "gp4d" 分區
#SBATCH --account=GOV113021          # 使用 "GOV113021" 計算資源帳戶
#SBATCH --ntasks-per-node=1          # 每個節點只執行 1 個任務
#SBATCH --cpus-per-task=4            # 每個任務分配 4 個 CPU 核心
#SBATCH --gpus-per-node=1            # 每個節點分配 1 個 GPU
#SBATCH --time=4-00:00:00            # 設定最大執行時間為 4 天
#SBATCH --output=logs/job-%j.out     # 標準輸出日誌文件，%j 代表作業 ID
#SBATCH --error=logs/job-%j.err      # 錯誤輸出日誌文件，%j 代表作業 ID
#SBATCH --mail-type=ALL              # 作業狀態變化時，發送所有通知
#SBATCH --mail-user=summerhill001@gmail.com  # 通知信箱

# VRAM:	14391MiB / 32768MiB 
# BS:	256
# Epoch: 2
# Dataset: 50k
# Max_seq_length: 8192
#  2/390 [04:11<13:28:29, 125.02s/it]

# 使用方式：
# 1. 提交作業: sbatch slurm_job/job_llama-3.2-3B-base.slurm
# 2. 監控作業日誌: 
#    tail -f logs/job-<job_id>.err  # 查看錯誤輸出
#    tail -f logs/job-<job_id>.out  # 查看標準輸出

# 載入 Singularity 模組
ml singularity

# 提示使用者下載 Singularity 映像檔
# 這一步需要手動執行一次: singularity pull docker://uptospace/unsloth:latest


# 建立虛擬專屬目錄
myBasedir="/work/c00cjz00/github/hpc_unsloth"
myHome="myhome/home_llama-3.2-3B-base"
mkdir -p ${myBasedir}/${myHome}

# 啟動訓練
singularity exec \
	--nv \
	--no-home \
	-B /work \
	-B ${myBasedir}/notebook:/notebook \
	-B ${myBasedir}/${myHome}:$HOME \
	${myBasedir}/unsloth-dev_latest.sif \
	bash -c "cd ~/; pwd; python3 /notebook/llama-3.2-3B-base.py"



