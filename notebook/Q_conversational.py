from unsloth import FastLanguageModel
import os

max_seq_length = 4096  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"/home/{os.getenv('USER')}/model",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    attn_implementation="default",   
)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are GENAI, an AI assistant created by NCHC. Provide clear, accurate, and helpful responses to the following instruction."},
    {"role": "user", "content": "台灣何時獨立"}
]

#Chage chat template to alpaca
# from unsloth.chat_templates import get_chat_template
# tokenizer = get_chat_template(
    # tokenizer,
    # chat_template = "alpaca", # Supports llama3, llama-3.1, phi-3, phi-3.5, phi-4, qwen-2.5, gemma, gemma, zephyr, chatml, mistral, llama, alpaca, unsloth
    # mapping = {"role":"from", "content":"value","user":"human","assistant":"gpt"}
# )


inputs = tokenizer.apply_chat_template ( messages, tokenize = True, add_generation_prompt = True, return_tensors = 'pt' ).to("cuda")
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024, use_cache = True)


