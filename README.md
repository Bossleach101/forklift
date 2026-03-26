# Neural Lifting and Deobfuscation

Repository extending the original *Forklift: An Extensible Neural Lifter* (COLM 2024) to target binary deobfuscation and assembly-to-IR translation. 

Forklift leverages language modeling to translate compiled assembly back to LLVM IR. In this extended work, we apply Forklift's architecture towards lifting and recovering semantics from intentionally obfuscated binaries, building a robust pipeline for evaluating structural and functional translation metrics on complex LLVM 14 toolchains.

## Datasets

### Obfuscated ExeBench
Our primary dataset, `leachl/obfuscated-exebench`, is an augmented form of the ExeBench dataset that combines obfuscated LLVM IR, assembly pairs, and execution `io_pairs` required for functional evaluations.

It is available on the Hugging Face Hub: https://huggingface.co/datasets/leachl/obfuscated-exebench

Usage via `datasets`:
```python
from datasets import load_dataset

# Supports streaming for large-scale evaluation
ds = load_dataset('leachl/obfuscated-exebench', split='train', streaming=True)
```

### Base ExeBench
The base clean dataset from the original Forklift paper is available at: https://huggingface.co/datasets/jordiae/exebench/tree/clang

## Pipeline & Evaluation

The repository includes tools for end-to-end evaluation of generated LLVM IR:

- **IR Syntax Validation (`forklift/ir_checker.py`)**: Uses intelligent heuristic-based fallbacks to recover undefined variables and function declarations to strict LLVM 14 typing standards.
- **SLURM Cluster Evaluation (`scripts/recheck_validity_job.sh`)**: High-throughput jobs for checking the translation syntax mapping rate and recompilation validity.
- **Neural Deobfuscation (`neurel_deob/`)**: Modules focused on processing and extracting semantics out of mangled instruction graphs.

## Model
See an example of the model architecture in `interactive.py`. Note that this represents a stripped down version of the primary lifting network used for demonstration.

## Acknowledgments

We would like to acknowledge that the files in the `forklift/` directory are derived from the original Forklift repository, with the exception of `forklift/ir_checker.py`, which was newly developed for this project.

## Original Paper

https://openreview.net/forum?id=LWfDcI6txJ#discussion

```bibtex
@inproceedings{
armengol-estape2024forklift,
title={Forklift: An Extensible Neural Lifter},
author={Jordi Armengol-Estap{\'e} and Rodrigo C. O. Rocha and Jackson Woodruff and Pasquale Minervini and Michael O'Boyle},
booktitle={First Conference on Language Modeling},
year={2024},
url={https://openreview.net/forum?id=LWfDcI6txJ}
}
```
