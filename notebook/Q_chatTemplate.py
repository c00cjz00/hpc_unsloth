from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
import os
model,tokenizer = FastLanguageModel.from_pretrained (
        model_name = f"/home/{os.getenv('USER')}/model",
        #model_name = "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit",
        max_seq_length = 2048,
        load_in_4bit = False,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "alpaca", # Supports llama3, llama-3.1, phi-3, phi-3.5, phi-4, qwen-2.5, gemma, gemma, zephyr, chatml, mistral, llama, alpaca, unsloth
    mapping = {"role":"from", "content":"value","user":"human","assistant":"gpt"}
)

FastLanguageModel.for_inference(model)

messages = [{"from":"human","value": "Give me a short introduction to large language model"}]
inputs = tokenizer.apply_chat_template ( messages, tokenize = True, add_generation_prompt = True, return_tensors = 'pt' ).to("cuda")
text_streamer = TextStreamer (tokenizer)
outputs = model.generate (input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024, use_cache = True )


