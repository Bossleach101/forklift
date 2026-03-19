import os
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizers import Tokenizer
from forklift.par_data import DP
from neurel_deob.training.data import DataConfig, ExeBenchDataset, collate_fn

def main():
    model_dir = "checkpoints/arm_ir_ir_v3/step_16000"
    tok_path = os.path.join(model_dir, "tokenizer.json")
    
    if os.path.exists(tok_path):
        print(f"Loading tokenizer from {tok_path}...")
        tok = Tokenizer.from_file(tok_path)
    else:
        print(f"Tokenizer not found at {tok_path}, falling back to HuggingFace base model...")
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained("jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b")
        tok = hf_tok._tokenizer # Get the underlying Rust Tokenizer object
    dp = DP(tokenizer=tok)
    
    cfg = DataConfig(
        hf_dataset="jordiae/exebench",
        split="test_synth",
        pair="arm_ir-ir",
        streaming=True,
        strip_ir_declares=True,
        normalize_ir_structs=True
    )
    
    print("Initializing dataset stream...")
    ds = ExeBenchDataset(
        tokenizer=tok,
        config=cfg,
        split="test_synth"
    )
    
    print("Fetching 2 valid samples from data stream...")
    samples = []
    for row in ds:
        samples.append(row)
        if len(samples) >= 2:
            break
            
    print("Collating batch...")
    pad_id = tok.get_vocab()["<pad>"]
    batch = collate_fn(samples, pad_id=pad_id, label_pad_id=-100)
    
    print("\n================ BATCH DIAGNOSTICS ================")
    print(f"input_ids shape:         {batch['input_ids'].shape}")
    print(f"attention_mask shape:    {batch['attention_mask'].shape}")
    print(f"labels shape:            {batch['labels'].shape}")
    print(f"decoder_input_ids shape: {batch['decoder_input_ids'].shape}")
    print("===================================================\n")
    
    for i in range(len(samples)):
        print(f"---------------- SAMPLE {i} -----------------")
        
        in_ids = batch['input_ids'][i].tolist()
        in_ids = [tid for tid in in_ids if tid != pad_id]
        in_text = dp.detokenize(in_ids)
        print(f"\n[INPUT_IDS (ARM Assembly) | {len(in_ids)} tokens]")
        print(in_text[:800] + ("\n... [TRUNCATED]" if len(in_text) > 800 else ""))
        
        lbl_ids = batch['labels'][i].tolist()
        lbl_ids = [tid for tid in lbl_ids if tid != -100]
        lbl_text = dp.detokenize(lbl_ids)
        print(f"\n[LABELS (LLVM IR Ground Truth) | {len(lbl_ids)} tokens]")
        print(lbl_text[:800] + ("\n... [TRUNCATED]" if len(lbl_text) > 800 else ""))
        
        dec_in_ids = batch['decoder_input_ids'][i].tolist()
        dec_in_ids = [tid for tid in dec_in_ids if tid != pad_id]
        dec_text = (dp.detokenize(dec_in_ids) if len(dec_in_ids) > 0 else "")
        print(f"\n[DECODER_INPUT_IDS | {len(dec_in_ids)} tokens]")
        print(dec_text[:200] + ("\n... [TRUNCATED]" if len(dec_text) > 200 else ""))
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
