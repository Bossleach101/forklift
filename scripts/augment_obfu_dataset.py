#!/usr/bin/env python3
"""
Augment an existing obfuscated dataset with missing ExeBench metadata fields.

This script pulls necessary functional verification columns (`c_deps`, `io_pairs`, etc.)
from the source ExeBench dataset (`jordiae/exebench`) and merges them into
your `leachl/obfuscated-exebench` dataset by matching `(exebench_split, fname)`.

It does NOT regenerate the obfuscated code. It simply acts as a SQL-style
LEFT JOIN to update the metadata, preserving all training data.

Usage:
    python scripts/augment_obfu_dataset.py \
        --dataset leachl/obfuscated-exebench \
        --hf-token <YOUR_WRITE_TOKEN> \
        --push-to-hub
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from typing import Dict, Any, Set, Tuple

from datasets import load_dataset, DatasetDict, Features, Value

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_metadata(
    source_repo: str,
    revision: str,
    needed_keys: Dict[str, Set[str]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Stream ExeBench and retrieve metadata for requested fnames.
    
    Args:
        source_repo: "jordiae/exebench"
        revision: "clang"
        needed_keys: Dict mapping {split_name: {set_of_fnames}}
    
    Returns:
        Dict mapping (split_name, fname) -> metadata_dict
    """
    lookup = {}
    total_found = 0
    
    for split_name, fnames in needed_keys.items():
        logger.info(f"Scanning source {source_repo} split='{split_name}' for {len(fnames)} fnames...")
        
        try:
            # Note: ExeBench requires subsets=[split_name] to load correctly in streaming mode
            # datasets < 2.16 produces error with trust_remote_code=True
            ds = load_dataset(
                source_repo, 
                split=split_name, 
                revision=revision, 
                subsets=[split_name], 
                streaming=True,
                # trust_remote_code=True # Removed for datasets 2.15 compatibility
            )
        except Exception as e:
            logger.warning(f"Could not load split '{split_name}' from {source_repo}: {e}")
            continue

        split_found = 0
        for row in ds:
            fname = row.get("fname")
            if fname in fnames:
                # Found a match! Extract metadata
                io_pairs = row.get("synth_io_pairs")
                if io_pairs is not None:
                    # Serialize to JSON string to store in Parquet
                    io_pairs = json.dumps(io_pairs)
                else:
                    io_pairs = ""

                meta = {
                    "c_deps": row.get("c_deps") or "",
                    "func_c_signature": row.get("func_c_signature") or "",
                    "cpp_wrapper": row.get("cpp_wrapper") or "",
                    "dummy_funcs": row.get("dummy_funcs") or "",
                    "io_pairs": io_pairs,
                }
                
                lookup[(split_name, fname)] = meta
                split_found += 1
                
                # Optimization: if we found all needed for this split, break early?
                # ExeBench fnames should be unique per split, but streaming gives no length info easily.
                # We'll just scan.
                if split_found >= len(fnames):
                    break
        
        logger.info(f"  Found {split_found}/{len(fnames)} records in '{split_name}'.")
        total_found += split_found

    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Target dataset (e.g. leachl/obfuscated-exebench)")
    parser.add_argument("--source-dataset", default="jordiae/exebench", help="Source dataset for metadata")
    parser.add_argument("--source-revision", default="clang", help="ExeBench revision (default: clang)")
    parser.add_argument("--push-to-hub", action="store_true", help="Push updated dataset to Hub")
    parser.add_argument("--hf-token", help="Hugging Face write token (optional if logged in)")
    parser.add_argument("--num-proc", type=int, default=4, help="Number of processes for mapping")
    args = parser.parse_args()

    # 1. Load the target dataset information (all splits)
    logger.info(f"Loading target dataset: {args.dataset}")
    target_ds = load_dataset(args.dataset)  # Returns DatasetDict

    # 2. Collect all (split, fname) pairs that need metadata
    # We group by source split name to efficiently stream ExeBench
    needed_keys: Dict[str, Set[str]] = defaultdict(set)
    
    total_rows = 0
    for split in target_ds:
        logger.info(f"Analyzing keys in target split: {split}")
        # We can iterate the dataset quickly to get keys
        # Assuming 'exebench_split' column exists. OLD generated datasets might rely 
        # on config defaults? Generator puts 'exebench_split' in column.
        
        # Check if column exists
        if "exebench_split" not in target_ds[split].column_names:
            logger.error(f"Column 'exebench_split' missing in {split}. Cannot join without knowing source split.")
            return

        # Use efficient batch selection if possible, or just iterate
        # Iterating a million rows is fast enough in python
        for row in target_ds[split]:
            src_split = row.get("exebench_split")
            fname = row.get("fname")
            if src_split and fname:
                needed_keys[src_split].add(fname)
            total_rows += 1

    logger.info(f"Identified {total_rows} rows requiring metadata across {len(needed_keys)} source splits.")

    # 3. Retrieve metadata from source
    lookup_table = fetch_metadata(args.source_dataset, args.source_revision, needed_keys)

    # 4. Map the dataset to add new columns
    logger.info("Merging metadata into target dataset...")

    def merge_metadata(batch):
        # Initialize output columns
        c_deps = []
        sigs = []
        wrappers = []
        dummies = []
        ios = []
        
        for i in range(len(batch["fname"])):
            fname = batch["fname"][i]
            src_split = batch["exebench_split"][i]
            
            meta = lookup_table.get((src_split, fname), {})
            
            c_deps.append(meta.get("c_deps", ""))
            sigs.append(meta.get("func_c_signature", ""))
            wrappers.append(meta.get("cpp_wrapper", ""))
            dummies.append(meta.get("dummy_funcs", ""))
            ios.append(meta.get("io_pairs", ""))
            
        return {
            "c_deps": c_deps,
            "func_c_signature": sigs,
            "cpp_wrapper": wrappers,
            "dummy_funcs": dummies,
            "io_pairs": ios
        }

    updated_ds = target_ds.map(
        merge_metadata,
        batched=True,
        num_proc=args.num_proc,
        desc="Augmenting dataset"
    )

    # 5. Push or Save
    if args.push_to_hub:
        logger.info(f"Pushing updated dataset to {args.dataset}...")
        updated_ds.push_to_hub(args.dataset, token=args.hf_token)
        logger.info("Success! Dataset updated on Hub.")
    else:
        logger.info("Dry run complete. Use --push-to-hub to save changes.")
        # print first row to show it worked
        first_split = list(updated_ds.keys())[0]
        print("Sample row:", updated_ds[first_split][0])

if __name__ == "__main__":
    main()
