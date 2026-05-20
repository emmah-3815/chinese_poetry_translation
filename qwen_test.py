# # from unsloth import FastLanguageModel
# # import torch

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "Qwen/Qwen2.5-14B-Instruct",
#     max_seq_length = 2048,
#     load_in_4bit = True, # QLoRA for 24GB VRAM or less
#     use_gradient_checkpointing = "unsloth",
#     attn_implementation = "flash_attention_2",
# )

# # A quick test for poetry
# inputs = tokenizer(
#     ["<|im_start|>user\nTranslate to English: 晚风拂柳笛声残，夕阳山外山。<|im_end|>\n<|im_start|>assistant\n"],
#     return_tensors = "pt"
# ).to("cuda")

# outputs = model.generate(**inputs, max_new_tokens = 64)
# print(tokenizer.decode(outputs[0]))

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

import sys
import time
import threading

def spinning_cursor():
    """A simple function to animate a spinner in the terminal."""
    chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    while getattr(threading.current_thread(), "do_run", True):
        for char in chars:
            sys.stdout.write(f'\r{char} Qwen is thinking...')
            sys.stdout.flush()
            time.sleep(0.1)
    # Clear the line when finished
    sys.stdout.write('\r' + ' ' * 30 + '\r')

os.environ["HF_TOKEN"] = "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"

model_name = "Qwen/Qwen2.5-14B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Keep the top part (model loading) the same...

messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:


    # 4. Now print the actual response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Qwen: {response}")

    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # 1. Add user message to history
    messages.append({"role": "user", "content": user_input})

    # 2. Prepare inputs
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 3. Generate response
    # --- Inside your main generation block ---

    # 1. Start the animation in a background thread
    spinner_thread = threading.Thread(target=spinning_cursor)
    spinner_thread.do_run = True
    spinner_thread.start()

    try:
        # 2. Run your model generation
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
    finally:
        # 3. Stop the animation as soon as generation finishes
        spinner_thread.do_run = False
        spinner_thread.join()
    # generated_ids = model.generate(**model_inputs, max_new_tokens=512)

    # 4. Decode just the new tokens
    new_tokens = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    # 5. Print and SAVE to history so the model remembers the context
    print(f"\nQwen: {response}")
    messages.append({"role": "assistant", "content": response})
