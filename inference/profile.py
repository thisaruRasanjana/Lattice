import time
import argparse
import mlx.core as mx
from mlx_lm import load, generate

def profile_model(model_name: str):
    print(f"\n{'='*60}")
    print(f"Profiling: {model_name}")
    print(f"{'='*60}")
    
    # Load model and tokenizer
    print("Loading model... (this may take a moment to download weights)")
    start_load = time.time()
    model, tokenizer = load(model_name)
    print(f"Loaded in {time.time() - start_load:.2f}s")
    
    # Prompts of different lengths
    prompts = {
        "Short (10 words)": "What is the capital of France and its population?",
        "Medium (~50 words)": "Explain the concept of Conway's Game of Life in simple terms. What are the rules and why is it considered computationally interesting?",
        "Long (~150 words)": "Write a detailed explanation of the difference between a CPU and a GPU. Discuss memory bandwidth, cache sizes, arithmetic logic units, and how their architectures dictate their ideal workloads. Use analogies to explain the concepts clearly to a beginner. Explain why matrix multiplication is better suited for GPUs while conditional logic is better suited for CPUs."
    }
    
    # Test Time To First Token (Prefill phase - compute bound)
    print("\n--- 1. Time To First Token (Prefill Phase) ---")
    for name, prompt in prompts.items():
        mx.reset_peak_memory()
        
        # tokenize
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # We only care about TTFT, so we generate exactly 1 token
        start_gen = time.time()
        _ = generate(model, tokenizer, prompt=prompt_text, max_tokens=1, verbose=False)
        ttft = time.time() - start_gen
        
        mem_gb = mx.get_peak_memory() / (1024**3)
        print(f"{name:<20}: TTFT = {ttft*1000:>6.1f} ms | Peak Mem = {mem_gb:.2f} GB")

    
    # Test Tokens/sec and KV Cache Growth (Decode phase - memory bound)
    print("\n--- 2. Decode Speed & KV-Cache Growth ---")
    base_prompt = "Write a comprehensive essay about the history of artificial intelligence."
    messages = [{"role": "user", "content": base_prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    gen_lengths = [10, 50, 100, 250, 500]
    
    for length in gen_lengths:
        mx.reset_peak_memory()
        
        start_gen = time.time()
        response = generate(model, tokenizer, prompt=prompt_text, max_tokens=length, verbose=False)
        total_time = time.time() - start_gen
        
        mem_gb = mx.get_peak_memory() / (1024**3)
        tok_sec = length / total_time
        
        print(f"Gen {length:>3} tokens: Speed = {tok_sec:>5.1f} tok/s | Peak Mem = {mem_gb:.3f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile LLM inference on Apple Silicon via MLX")
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["mlx-community/Qwen2.5-0.5B-Instruct-4bit", 
                                 "mlx-community/Qwen2.5-1.5B-Instruct-4bit"],
                        help="HuggingFace model repos to profile")
    args = parser.parse_args()
    
    for m in args.models:
        profile_model(m)
