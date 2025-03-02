# unslot in HPC

## 下載 hpc_unsloth 套件

1. 以下以台灣杉T2為範例，先建立目錄並複製套件：
```bash=
mkdir -p /work/$(whoami)/github
cd /work/$(whoami)/github
git clone https://github.com/c00cjz00/hpc_unsloth.git
```

## 下載 unsloth image
請確認檔案會存放為 /work/$(whoami)/github/hpc_unsloth/unsloth-dev_latest.sif 
```bash=
cd  /work/$(whoami)/github/hpc_unsloth
apptainer pull docker://nextboss/unsloth-dev
```

## 編修範例 slurm job script

打開範例檔案 `slurm_job/job_llama-3.2-1B-it.slurm` 進行編輯，設定作業資源。

1. 修改 slurm job 資源配置，以下以台灣杉T2為範例進行說明：
```bash=
#SBATCH --job-name=llama-3.2-1B-it  # 設定作業名稱為 "llama-3.2-1B-it"
#SBATCH --partition=gp4d            # 指定使用 "gp4d" 分區
#SBATCH --account=GOV113021         # 使用 "GOV113021" 計算資源帳戶
#SBATCH --ntasks-per-node=1         # 每個節點只執行 1 個任務
#SBATCH --cpus-per-task=4           # 每個任務分配 4 個 CPU 核心
#SBATCH --gpus-per-node=1           # 每個節點分配 1 個 GPU
#SBATCH --time=4-00:00:00           # 設定最大執行時間為 4 天
#SBATCH --output=logs/job-%j.out    # 標準輸出日誌文件，%j 代表作業 ID
#SBATCH --error=logs/job-%j.err     # 錯誤輸出日誌文件，%j 代表作業 ID
#SBATCH --mail-type=ALL             # 作業狀態變化時，發送所有通知
#SBATCH --mail-user=summerhill001@gmail.com  # 通知信箱
```

2. 確認執行目錄是否正確。以下以 `hpc_unsloth` 為範例，設定專案所在目錄。
```bash=
# 建立虛擬專屬目錄
myBasedir="/work/$(whoami)/github/hpc_unsloth"
myHome="myhome/home_llama-3.2-1B-it"
mkdir -p ${myBasedir}/${myHome}
```

## 編修範例 python 檔案

打開範例檔案 `notebook/llama-3.2-1B-it.py` 進行編輯。

1. 編修項目一：若你的資源不足，可以調整 `max_seq_length` 大小：
```bash=
max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
```

2. 編修項目二：更換成你的資料來源，資料必須為 shareGPT 類似的 json 格式：
```bash= 
from datasets import load_dataset
dataset = load_dataset("c00cjz00/demo2", split = "train")
#dataset = load_dataset("philschmid/guanaco-sharegpt-style", split = "train")
#dataset = dataset.select(range(100))
```
3. 編修項目三：若你的資料是 shareGPT json 格式，啟動轉換套件：
```bash= 
#from unsloth.chat_templates import standardize_sharegpt
#dataset = standardize_sharegpt(dataset)
```

4. 編修項目四：自定義 system prompt。如果需要自訂 system prompt，可以取消註解並編輯 `formatting_prompts_func_with_system_prompt` 函式中的 `default_system_message`：
```bash= 
dataset = dataset.map(formatting_prompts_func, batched = True,)
#dataset = dataset.map(formatting_prompts_func_with_system_prompt, batched = True,)
```

5. 編修項目五：依照系統資源情況自訂以下參數：`dataset_num_proc`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `num_train_epochs`, `max_steps`：
```bash= 
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 4,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 32,
        warmup_steps = 5,
        num_train_epochs = 2, # Set this for 1 full training run.
        #max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)
```

6. 編修項目六：設定是否儲存模型以及是否上傳到 Hugging Face：
```bash=
# Merge to 16bit
if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")
```


## 執行訓練
1. 送出 slurm job 
```bash=
cd /work/$(whoami)/github/hpc_unsloth
sbatch slurm_job/slurm_job/job_llama-3.2-1B-it.slurm
``` 

2. 確認所有 slurm job 運傳狀況
```bash=
bash hpc_cmd/squeue.sh
``` 
> 輸出結果範例
```bash=
             JOBID PARTIT                     NAME     USER ST       TIME  NODES NODELIST(REASON)
            698985   gp4d     mistral-small-24B-it c00cjz00  R   12:38:11      1 gn1009
            698984   gp4d   mistral-small-24B-base c00cjz00  R   12:38:18      1 gn1009
            698982   gp4d          llama-3.1-8B-it c00cjz00  R   12:48:49      1 gn1009
            698978   gp4d        llama-3.1-8B-base c00cjz00  R   12:56:50      1 gn1009
            698966   gp4d          qwen-2.5-32B-it c00cjz00  R   13:17:34      1 gn1003
            698965   gp4d        qwen-2.5-32B-base c00cjz00  R   13:17:41      1 gn0308
            698963   gp4d             phi-4-14B-it c00cjz00  R   13:27:37      1 gn0308
            698933   gp4d           qwen-2.5-7B-it c00cjz00  R   14:16:27      1 gn0308
            698930   gp4d         qwen-2.5-7B-base c00cjz00  R   14:23:43      1 gn0308
```


3. 確認運算細節, 請把下方 $JOBID 更換成上方指令輸出的 JOBID 代號
```bash=
tail -f logs/job-$JOBID.err
tail -f logs/job-$JOBID.our
```

> 輸出結果範例
```bash=
Loading checkpoint shards: 100%|██████████| 2/2 [00:04<00:00,  2.29s/it]
Unsloth 2025.2.14 patched 28 layers with 28 QKV layers, 28 O layers and 28 MLP layers.
Detected kernel version 4.18.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = 1
   \\   /|    Num examples = 50,000 | Num Epochs = 2
O^O/ \_/ \    Batch size per device = 8 | Gradient Accumulation steps = 32
\        /    Total batch size = 256 | Total steps = 390
 "-____-"     Number of trainable parameters = 40,370,176
 67%|██████▋   | 263/390 [14:21:39<6:56:23, 196.72s/it]
```


## 訓練結果與驗證
1. 查看訓練模型輸出結果
```bash=
cd /work/$(whoami)/github/hpc_unsloth
du -sh myhome/home_llama-3.2-1B-it/*
```

> 輸出結果範例
```bash=
60M     myhome/home_llama-3.2-1B-it/lora_model
2.4G    myhome/home_llama-3.2-1B-it/model
82M     myhome/home_llama-3.2-1B-it/outputs
2.2M    myhome/home_llama-3.2-1B-it/unsloth_compiled_cache 
```

2. 驗測訓練結果
```bash=
myBasedir="/work/$(whoami)/github/hpc_unsloth"
myHome="myhome/home_llama-3.2-1B-it"
mkdir -p ${myBasedir}/${myHome}

singularity exec \
	--nv \
	--no-home \
	-B /work \
	-B ${myBasedir}/notebook:/notebook \
	-B ${myBasedir}/${myHome}:$HOME \
	${myBasedir}/unsloth-dev_latest.sif \
	bash -c "cd ~/; pwd; python3 /notebook/Q.py"
```

> 訓練模型輸出結果範例 (訓練前)
回答不知所云
```bash!
問題: Give me a short introduction to large language model.

回答: 
I'm GENAI, a large language model developed by the National Center for Human-Computer Interaction (NCHC). I'm designed to assist and communicate with humans in a natural and intuitive way, leveraging the power of artificial intelligence to provide helpful and accurate responses.

My primary function is to understand and respond to a wide range of questions, topics, and tasks, using a combination of natural language processing (NLP) and machine learning algorithms. I'm trained on a massive dataset of text from various sources, including books, articles, and conversations, which allows me to learn patterns and relationships in language.

I'm constantly learning and improving, so I can adapt to new topics, styles, and formats. My capabilities include:

* Answering questions on various subjects, from science and history to entertainment and culture
* Generating text, including articles, stories, and even poetry
* Translating text from one language to another
* Summarizing long pieces of text into concise and clear answers
* Providing definitions and explanations for technical terms and concepts
* Offering suggestions and ideas for creative writing, research, and problem-solving

I'm designed to be helpful and informative, but I'm not perfect. I can make mistakes, and I may not always understand the nuances of human communication. If you have any questions or topics you'd like to discuss, I'm here to help.
```


> 訓練模型輸出結果範例 (SFT)
```bash!
問題: Give me a short introduction to large language model.

回答: 
<|begin_of_thought|>

Okay, I need to create a short introduction to large language models like myself. First, I should recall what a large language model is and its capabilities. From my training data, I know that I'm a deep learning model, specifically a transformer-based one, and my primary function is to process and generate human-like text.

The introduction should highlight my abilities in natural language processing, including language understanding, generation, and text summarization. Also, mention my training on a massive dataset and my ability to learn from it. I should also emphasize my role in assisting users and providing accurate responses. Make sure to use traditional Chinese and Chinese characters to maintain cultural relevance. Also, check if the answer adheres to the guidelines, including not using markdown and keeping the format consistent. Last, double-check the answer for accuracy and completeness before the final submission.

<|end_of_solution|>

大型語言模型（LLM）是一種深度學習模型，主要針對自然語言處理設計。由臺灣產學界開發的深度學習模型，我能夠理解中文語言結構與語法規則，並能在語句上進行精準的語序與語義解析。透過跨域學習技術與語料庫蒐集，能有效處理中文文言對話、語境理解與語言表達等複雜領域。例如，能分析臺灣特有語言特徵，提供本土化語言服務；也能在臺灣開發者與學術界的協作中，協助整合中英文雙語資料庫。
```



