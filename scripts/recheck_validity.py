#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from forklift.ir_checker import check_ir, IRCheckResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def check_single(item):
    """
    Check a single prediction item.
    item is a dict from the jsonl file.
    """
    pred = item.get("prediction", "")
    fname = item.get("fname", "unknown")
    
    # Run the check (Level 1 syntax, maybe Level 2 compile if configured in check_ir default)
    # We want to check syntax primarily to confirm the fix.
    # Level 2 (compilation) requires clang. 
    res = check_ir(pred, level=2, auto_declare=True)
    
    return {
        "fname": fname,
        "syntax_valid": res.syntax_valid,
        "compiles": res.compiles,
        "syntax_error": res.syntax_error,
        "compile_error": res.compile_error
    }

def main():
    parser = argparse.ArgumentParser(description="Re-check validity of predictions jsonl")
    parser.add_argument("input_file", help="Path to preds_Test_....jsonl file")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--output", help="Optional output JSON file for results")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    logger.info(f"Loading predictions from {args.input_file}...")
    items = []
    with open(args.input_file, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    logger.info(f"Checking {len(items)} predictions with {args.workers} workers...")
    
    results = []
    syntax_pass = 0
    compile_pass = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(check_single, item): item for item in items}
        
        for future in tqdm(as_completed(futures), total=len(items)):
            res = future.result()
            results.append(res)
            
            if res["syntax_valid"]:
                syntax_pass += 1
            if res["compiles"]:
                compile_pass += 1

    total = len(items)
    syntax_rate = (syntax_pass / total) * 100 if total else 0
    compile_rate = (compile_pass / total) * 100 if total else 0

    logger.info("=" * 40)
    logger.info(f"Total processed: {total}")
    logger.info(f"Syntax Valid:    {syntax_pass} ({syntax_rate:.2f}%)")
    logger.info(f"Compiles:        {compile_pass} ({compile_rate:.2f}%)")
    logger.info("=" * 40)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "total": total,
                "syntax_valid": syntax_pass,
                "compiles": compile_pass,
                "syntax_rate": syntax_rate,
                "compile_rate": compile_rate,
                "details": results
            }, f, indent=2)
        logger.info(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main()
