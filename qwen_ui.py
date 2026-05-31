"""
poetry_ui.py
────────────
Interactive terminal interface for classical Chinese → English poetry translation.
Works with the base Qwen2.5-1.5B-Instruct model OR a QLoRA-fine-tuned adapter
produced by train_qwen_poetry.py.

Usage
-----
# Base model only
    python poetry_ui.py

# With your fine-tuned LoRA adapter
    python poetry_ui.py --adapter ./qwen_poetry_lora

# Side-by-side comparison (base vs fine-tuned)
    python poetry_ui.py --adapter ./qwen_poetry_lora --compare

Options
-------
  --model     HuggingFace model id   (default: Qwen/Qwen2.5-1.5B-Instruct)
  --adapter   Path to LoRA adapter   (optional)
  --compare   Show base vs LoRA side-by-side
  --no_4bit   Disable 4-bit quant    (use if you have >8 GB free)
  --max_new   Max new tokens          (default: 256)
  --hf_token  HuggingFace token      (or set HF_TOKEN env var)
"""

import argparse
import os
import sys
import textwrap
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Pastel ANSI colours (256-colour palette) ──────────────────────────────────
# Uses ESC[38;5;<n>m for foreground, ESC[1m for bold, ESC[2m for dim
def _fg(n): return f"\033[38;5;{n}m"

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"

# Pastel ramp — soft 256-colour picks
LAVENDER  = _fg(183)   # soft purple   — prompts, headings
PEACH     = _fg(223)   # warm peach    — original poem / user input
SAGE      = _fg(151)   # muted sage    — fine-tuned output
ROSE      = _fg(218)   # dusty rose    — compare / toggles
SKY       = _fg(153)   # pale sky blue — spinner / accents
SAND      = _fg(187)   # warm sand     — dim metadata
BLUSH     = _fg(210)   # soft blush    — error / warnings

# Aliases kept for backward-compat with the rest of the file
CYAN    = SKY
YELLOW  = PEACH
GREEN   = SAGE
MAGENTA = ROSE
RED     = BLUSH

def c(text, *codes): return "".join(codes) + str(text) + RESET

# ── System prompt (same as training) ─────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a literary translator specialising in classical Chinese poetry. "
    "Translate the given Chinese poem into elegant English verse, preserving "
    "its meaning, imagery, and poetic quality."
)

# ── Spinner ───────────────────────────────────────────────────────────────────
class Spinner:
    FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, label="Translating"):
        self.label = label
        self._thread = None
        self._stop   = threading.Event()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        sys.stdout.write("\r" + " " * (len(self.label) + 6) + "\r")
        sys.stdout.flush()

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            sys.stdout.write(f"\r{c(self.FRAMES[i % len(self.FRAMES)], LAVENDER)}  {c(self.label + '…', SAND)}")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_name: str, adapter_path: str | None,
               use_4bit: bool, hf_token: str | None):
    """Load tokeniser + model (optionally with LoRA adapter)."""

    print(c(f"\n  Loading {model_name} …", DIM))

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        from peft import PeftModel
        print(c(f"  Attaching LoRA adapter from {adapter_path} …", DIM))
        tuned = PeftModel.from_pretrained(base, adapter_path)
        tuned.eval()
        return tokenizer, base, tuned   # base + tuned both available
    else:
        base.eval()
        return tokenizer, base, None

# ── Inference ─────────────────────────────────────────────────────────────────
def translate(model, tokenizer, poem: str,
              title: str = "", author: str = "",
              max_new_tokens: int = 256) -> tuple[str, dict]:
    """
    Run a single translation.
    Returns (translation_text, perf) where perf contains:
      - elapsed_s   : total wall time in seconds
      - elapsed_ms  : total wall time in milliseconds
      - tokens      : number of new tokens generated
      - tok_per_s   : tokens per second
      - peak_vram_gb: peak VRAM during generation (GPU only)
    """
    import time

    parts = []
    if title and author:
        parts.append(f"Poem: \u300a{title}\u300b by {author}")
    elif title:
        parts.append(f"Poem: \u300a{title}\u300b")
    parts.append(f"Translate the following classical Chinese poem into English:\n\n{poem}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "\n".join(parts)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_ids   = out[0][inputs["input_ids"].shape[-1]:]
    n_tokens  = len(new_ids)
    vram_gb   = (torch.cuda.max_memory_reserved() / 1024 ** 3
                 if torch.cuda.is_available() else 0.0)

    perf = {
        "elapsed_s":    round(elapsed, 3),
        "elapsed_ms":   round(elapsed * 1000, 1),
        "tokens":       n_tokens,
        "tok_per_s":    round(n_tokens / elapsed, 1) if elapsed > 0 else 0,
        "peak_vram_gb": round(vram_gb, 2),
    }
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, perf

# ── Display helpers ───────────────────────────────────────────────────────────
WIDTH = 72

def banner():
    print(c("·" * WIDTH, LAVENDER))
    print(c("  詩  Classical Chinese Poetry Translator", LAVENDER + BOLD))
    print(c("  ·  Powered by Qwen2.5-1.5B-Instruct + QLoRA", SAND))
    print(c("·" * WIDTH, LAVENDER))
    print()

def divider(label=""):
    if label:
        pad = (WIDTH - len(label) - 2) // 2
        print(c("·" * pad + f" {label} " + "·" * pad, LAVENDER))
    else:
        print(c("·" * WIDTH, SAND))

def print_poem_block(label: str, text: str, colour):
    divider(label)
    wrapped = textwrap.fill(text, width=WIDTH - 4)
    for line in wrapped.splitlines():
        print("  " + c(line, colour))
    print()

def show_help():
    print(c("""
  Commands
  ────────
  Just type (or paste) your Chinese poem and press Enter.
  For multi-line poems, end with a blank line.

  :title  <title>   Set poem title for richer context
  :author <author>  Set poem author
  :meta             Show current title / author
  :clear            Clear title and author
  :compare          Toggle base vs fine-tuned side-by-side
  :temp   <0-1>     Set sampling temperature  (default 0.7)
  :tokens <n>       Set max new tokens         (default 256)
  :help             Show this message
  :quit / :exit     Exit
""", DIM))

# ── Perf display ──────────────────────────────────────────────────────────────

def _print_perf(perf: dict):
    vram = f"  ·  VRAM {perf['peak_vram_gb']:.2f} GB" if perf["peak_vram_gb"] > 0 else ""
    print(c(
        f"  ⏱  {perf['elapsed_ms']:.0f} ms"
        f"  ·  {perf['tokens']} tokens"
        f"  ·  {perf['tok_per_s']:.1f} tok/s"
        f"{vram}",
        SAND
    ))


# ── Main REPL ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter",  default=None,
                        help="Path to QLoRA adapter (from train_qwen_poetry.py)")
    parser.add_argument("--compare",  action="store_true",
                        help="Show base model vs fine-tuned side by side")
    parser.add_argument("--no_4bit",  action="store_true")
    parser.add_argument("--max_new",  type=int, default=256)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    banner()

    tokenizer, base_model, tuned_model = load_model(
        args.model, args.adapter,
        use_4bit=not args.no_4bit,
        hf_token=args.hf_token,
    )

    has_adapter  = tuned_model is not None
    compare_mode = args.compare and has_adapter
    max_new      = args.max_new
    temperature  = 0.7

    # Session state
    title  = ""
    author = ""

    if has_adapter:
        print(c("  ✓ Fine-tuned adapter loaded", SAGE))
    else:
        print(c("  ✓ Base model loaded (no adapter)", PEACH))
    if compare_mode:
        print(c("  ✓ Compare mode ON  (base vs fine-tuned)", ROSE))
    print()
    print(c("  Type :help for commands, :quit to exit.", DIM))
    print()

    # ── Input loop ────────────────────────────────────────────────────────────
    while True:
        try:
            # Prompt line
            prefix = ""
            if title:  prefix += c(f"[{title}] ", YELLOW)
            if author: prefix += c(f"by {author}  ", DIM)
            sys.stdout.write(c("你 › ", LAVENDER + BOLD) + prefix)
            sys.stdout.flush()
            first_line = input()
        except (EOFError, KeyboardInterrupt):
            print(c("\n\n  再见！(Goodbye)\n", CYAN))
            break

        line = first_line.strip()

        # ── Commands ──────────────────────────────────────────────────────────
        if not line:
            continue

        if line.startswith(":"):
            parts = line.split(None, 1)
            cmd   = parts[0].lower()
            arg   = parts[1] if len(parts) > 1 else ""

            if cmd in (":quit", ":exit", ":q"):
                print(c("\n  再见！(Goodbye)\n", CYAN))
                break
            elif cmd == ":help":
                show_help()
            elif cmd == ":title":
                title = arg
                print(c(f"  Title set to: {title}", YELLOW))
            elif cmd == ":author":
                author = arg
                print(c(f"  Author set to: {author}", YELLOW))
            elif cmd == ":meta":
                print(c(f"  Title: {title or '(none)'}  |  Author: {author or '(none)'}", DIM))
            elif cmd == ":clear":
                title = author = ""
                print(c("  Title and author cleared.", DIM))
            elif cmd == ":compare":
                if not has_adapter:
                    print(c("  No adapter loaded — compare mode unavailable.", RED))
                else:
                    compare_mode = not compare_mode
                    state = "ON" if compare_mode else "OFF"
                    print(c(f"  Compare mode {state}.", MAGENTA))
            elif cmd == ":temp":
                try:
                    temperature = float(arg)
                    print(c(f"  Temperature set to {temperature}", DIM))
                except ValueError:
                    print(c("  Usage: :temp 0.7", RED))
            elif cmd == ":tokens":
                try:
                    max_new = int(arg)
                    print(c(f"  Max new tokens set to {max_new}", DIM))
                except ValueError:
                    print(c("  Usage: :tokens 256", RED))
            else:
                print(c(f"  Unknown command: {cmd}  (type :help)", RED))
            print()
            continue

        # ── Multi-line poem collection ─────────────────────────────────────────
        # If the user pastes multiple lines, collect until a blank line
        poem_lines = [line]
        while True:
            try:
                next_line = input()
            except (EOFError, KeyboardInterrupt):
                break
            if next_line.strip() == "":
                break
            poem_lines.append(next_line.strip())
        poem = "\n".join(poem_lines)

        print()
        print_poem_block("Original", poem, YELLOW)

        # ── Translate ─────────────────────────────────────────────────────────
        spinner = Spinner("Translating")

        if compare_mode:
            # Base model
            spinner.label = "Base model translating"
            spinner.start()
            base_out, base_perf = translate(base_model, tokenizer, poem, title, author, max_new)
            spinner.stop()
            print_poem_block("Base Model", base_out, DIM)
            _print_perf(base_perf)

            # Fine-tuned model
            spinner.label = "Fine-tuned model translating"
            spinner.start()
            tuned_out, tuned_perf = translate(tuned_model, tokenizer, poem, title, author, max_new)
            spinner.stop()
            print_poem_block("Fine-tuned (LoRA)", tuned_out, GREEN)
            _print_perf(tuned_perf)

        else:
            active = tuned_model if has_adapter else base_model
            label  = "Fine-tuned (LoRA)" if has_adapter else "Base Model"
            colour = GREEN if has_adapter else CYAN

            spinner.start()
            out, perf = translate(active, tokenizer, poem, title, author, max_new)
            spinner.stop()
            print_poem_block(label, out, colour)
            _print_perf(perf)

        divider()
        print()


if __name__ == "__main__":
    main()