# chinese_poetry_translation
ECE 175b project

# How To Get Started
Run these in your terminal to set up the environment
1. Create the environment
conda create --name qwen_poetry python=3.12 -y
2. Activate it
conda activate qwen_poetry

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

3. Core fine-tuning libraries
pip install unsloth "trl<0.12.0" peft accelerate bitsandbytes

4. Transformers and data handling
pip install transformers datasets sentencepiece protobuf


6. pip install unsloth_zoo

7. Optional: WandB for tracking your poetry translation metrics
pip install wandb

# Run the qwen model tester
python qwen_test.py