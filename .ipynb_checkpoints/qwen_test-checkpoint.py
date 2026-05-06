from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-14B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True, # QLoRA for 24GB VRAM or less
    use_gradient_checkpointing = "unsloth",
    attn_implementation = "sdpa",
)

# A quick test for poetry
inputs = tokenizer(
    ["<|im_start|>user\nTranslate to English: 晚风拂柳笛声残，夕阳山外山。<|im_end|>\n<|im_start|>assistant\n"],
    return_tensors = "pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64)
print(tokenizer.decode(outputs[0]))