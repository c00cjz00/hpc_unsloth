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

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are GENAI, created by NCHC. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024, use_cache = True)

