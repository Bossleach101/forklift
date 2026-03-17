import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

def main():
    base_model_name = "leachl/forklift-arm-ir-ir"
    checkpoint_dir = "leachl/forklift-arm-ir-ir-dpo"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    print(f"Loading model from {checkpoint_dir}...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Make sure you are running this from the root of the workspace.")
        sys.exit(1)
        
    model.eval()
    
    # A tiny arbitrary ARM assembly snippet to trigger generation
    sample_arm_prompt = """
    .text
    .align 2
    .global my_dummy_func
my_dummy_func:
    mov r0, #0
    bx lr
"""

    print("--- Input Prompt ---")
    print(sample_arm_prompt.strip())
    print("--------------------")
    
    inputs = tokenizer(sample_arm_prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    print("\nGenerating sequence... (checking for mode-collapse)")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n--- Model Output ---")
    if not decoded.strip():
        print("[EMPTY STRING - The model produced no output tokens!]")
    else:
        print(decoded)
    print("--------------------\n")

if __name__ == "__main__":
    main()
